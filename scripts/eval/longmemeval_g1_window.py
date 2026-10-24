"""CPU-only G1 input diagnosis; whitelist template/tokenizer endpoints, never inference."""
import hashlib
import math
from pathlib import Path
import time
import numpy as np
import longmemeval_g1_run as original

lm, mm, am, st = original.lm, original.mm, original.am, original.st
SOURCE = original.LOCAL
LOCAL = SOURCE.parent/'window-diagnosis'


def select_window(maximum):
    required = math.ceil(1.10*(maximum+120+128))
    return required, next((n for n in (16384, 32768, 65536) if n >= required), None)


class CPUCount:
    def __init__(self):
        self.url = st.Qwen.offline().settings['server']['url']
        self.counts = lm.read(SOURCE/'token-counts.json').copy()
        self.http_calls = 0

    def request(self, endpoint, payload):
        if endpoint not in ('/tokenize', '/apply-template'):
            raise RuntimeError('only CPU template/tokenizer endpoints allowed')
        self.http_calls += 1
        return st.http_json(self.url+endpoint, payload, timeout=900)

    def count(self, text):
        if not text:
            return 0
        key = lm.digest(text)
        if key not in self.counts:
            self.counts[key] = len(self.request('/tokenize', {'content': text,
                   'add_special': False, 'parse_special': False})['tokens'])
        return self.counts[key]

    def input_count(self, messages):
        prompt = self.request('/apply-template', {'messages': messages,
                     'chat_template_kwargs': am.GENERATION['chat_template_kwargs']})['prompt']
        return len(self.request('/tokenize', {'content': prompt, 'add_special': True,
                                             'parse_special': True})['tokens'])


def summary(values):
    return dict(zip(('min', 'p50', 'p90', 'max'), np.quantile(values, [0, .5, .9, 1]).tolist()))


def candidate_reference_check():
    """Additional CPU references only; no rerank output is assumed or produced."""
    path = lm.RESULTS/'longmemeval-g1-C-reference-lengths.json'
    if path.exists():
        raise RuntimeError('reference check exists; do not overwrite')
    prepared, jobs = lm.read(SOURCE/'prepared.json'), lm.read(LOCAL/'rerank-jobs.json')
    counter = CPUCount(); counter.counts.update(lm.read(LOCAL/'token-counts.json'))
    rows = []
    for qid, job in jobs.items():
        lengths = []
        for number in range(min(len(job['order']), mm.RERANK_CANDIDATES)):
            order = mm.distinct(original.base.norw.prod.reranked_order(job['order'], [number]), job['units'])
            raw_ids = list(dict.fromkeys(t for i in order for t in job['units'][i]['turns']))
            payload = am.payload('C-final-norw', prepared['turns'][qid], {'C-final-norw': raw_ids}, counter.count)
            chat = am.messages(payload['conversation'], payload['retrieved'], prepared['questions'][qid]['question'])
            lengths.append(counter.input_count(chat))
        rows.append({'id': qid, 'each_candidate_first_input_tokens': lengths, 'max_reference_input': max(lengths),
                     'all_reference_inputs_fit8192': all(original.fits(n, 256, 8192) for n in lengths)})
    lm.write(LOCAL/'token-counts.json', counter.counts)
    lm.write(path, {'rows': rows, 'max_reference_input': max(r['max_reference_input'] for r in rows),
        'all_reference_inputs_fit8192': all(r['all_reference_inputs_fit8192'] for r in rows),
        'scope': 'each of30 candidates first; not all possible permutations and not actual final C',
        'inference_HTTP_sent': 0, 'strict_actual_C_preflight_still_required': True})


def diagnose():
    if LOCAL.exists():
        raise RuntimeError('diagnosis exists; do not overwrite')
    LOCAL.mkdir(parents=True)
    p = lm.read(original.PLAN); original.verify(p)
    journal = [__import__('json').loads(s) for s in (SOURCE/'received.jsonl').read_text().splitlines()]
    if len(journal) != 1441 or any(not r['key'].startswith('fact|') for r in journal):
        raise RuntimeError('expected only saved ingestion; no evaluation outputs observed')
    if (SOURCE/'inflight.json').exists():
        raise RuntimeError('unknown inflight')
    prepared, embedding = lm.read(SOURCE/'prepared.json'), lm.read(SOURCE/'embedding.json')
    cfg = mm.config(['p2', 'fact'], rerank='qwen')
    counter = CPUCount(); jobs, rows, C_reference = {}, [], {}
    started = time.monotonic()
    try:
        for qid, question in prepared['questions'].items():
            ts, units = prepared['turns'][qid], embedding['units'][qid]
            vectors = np.load(SOURCE/f'doc-vectors-{qid}.npy', allow_pickle=False)
            index = mm.Index(units, vectors)
            start = time.monotonic()
            order = mm.distinct([i for i, _ in index.rank(cfg, qid,
                     [(np.asarray(embedding['query_vectors'][qid]), question['question'])])], units)
            search_seconds = time.monotonic()-start
            chat = original.base.norw.prod.rerank_chat({'text': question['question']}, order, units, ts, cfg)
            tokens = counter.input_count(chat)
            lookup = {t['id']: (n, t) for n, t in enumerate(ts)}
            candidates = []
            for number, i in enumerate(order[:mm.RERANK_CANDIDATES], 1):
                ids = sorted(units[i]['turns'], key=lambda tid: lookup[tid][0])
                body = '\n'.join(lookup[tid][1]['text'] for tid in ids)
                passage = f'[{number}] '+mm.dated(body, lookup[ids[0]][1]['date'], mm.uses_dates(cfg))
                candidates.append({'candidate_number': number, 'unit_id': units[i]['id'],
                    'original_turn_ids': ids, 'passage_tokens': counter.count(passage),
                    'single_candidate_input_tokens': counter.input_count(mm.messages('rerank',
                        question=question['question'], passages=passage))})
            first_crossing = None
            if not original.fits(tokens, 120, 8192):
                lo, hi = 1, len(candidates)
                while lo < hi:
                    mid = (lo+hi)//2
                    n = counter.input_count(original.base.norw.prod.rerank_chat(
                        {'text': question['question']}, order[:mid], units, ts, cfg))
                    if original.fits(n, 120, 8192): lo = mid+1
                    else: hi = mid
                first_crossing = lo
            raw_ids = list(dict.fromkeys(t for i in order for t in units[i]['turns']))
            payload = am.payload('C-final-norw', ts, {'C-final-norw': raw_ids}, counter.count)
            answer_chat = am.messages(payload.pop('conversation'), payload.pop('retrieved'), question['question'])
            C_reference[qid] = {'messages': answer_chat, 'input_tokens': counter.input_count(answer_chat),
                                **payload, 'scope': 'dense-order reference, not final reranked answer input'}
            # Structural budget estimate; actual reranked C gets strict preflight before answering.
            skeleton = counter.input_count(am.messages('', '', question['question']))
            envelope = skeleton+1024+512+128
            single_overflow = [c['candidate_number'] for c in candidates
                               if not original.fits(c['single_candidate_input_tokens'], 120, 8192)]
            rows.append({'id': qid, 'candidate_count': len(candidates), 'rerank_input_tokens': tokens,
                 'rerank_with_output_margin': tokens+120+128, 'overflows_8192': not original.fits(tokens, 120, 8192),
                 'max_candidate_tokens': max(c['passage_tokens'] for c in candidates),
                 'single_candidate_overflow_numbers': single_overflow, 'first_prefix_crossing_candidate': first_crossing,
                 'cause': 'single candidate alone exceeds8k' if single_overflow else
                          'combined unchanged candidates exceed8k' if tokens+248>8192 else 'fits8k',
                 'candidates': candidates,
                 'C_dense_reference_input_tokens': C_reference[qid]['input_tokens'],
                 'C_input_budget_envelope_estimate': envelope,
                 'C_envelope_plus_output_margin_estimate': envelope+256+128,
                 'C_dense_reference_fits8192': original.fits(C_reference[qid]['input_tokens'], 256, 8192),
                 'A_actual_input_tokens': prepared['inputs']['A'][qid]['input_tokens'],
                 'B_actual_input_tokens': prepared['inputs']['B1536'][qid]['input_tokens']})
            body = st.Qwen.offline().request('rerank', chat)
            jobs[qid] = {'body': body, 'input_tokens': tokens, 'order': order, 'search_seconds': search_seconds,
                         'body_sha256': lm.digest(body), 'units': units}
            print('input diagnosis', len(rows), '/30', flush=True)
        required, chosen = select_window(max(r['rerank_input_tokens'] for r in rows))
        answer_ok = all(r['C_dense_reference_fits8192'] and r['C_envelope_plus_output_margin_estimate'] <= 8192
                        and original.fits(r['B_actual_input_tokens'], 256, 8192) for r in rows)
        report = {'time': lm.stamp(), 'window_rule_commit': 'd2e7f4f', 'questions': 30,
                  'rerank_distribution': summary([r['rerank_input_tokens'] for r in rows]),
                  'C_dense_reference_distribution': summary([r['C_dense_reference_input_tokens'] for r in rows]),
                  'B_actual_distribution': summary([r['B_actual_input_tokens'] for r in rows]),
                  'A_actual_distribution': summary([r['A_actual_input_tokens'] for r in rows]),
                  'required_with10percent': required, 'selected_rerank_context': chosen,
                  'overflow8k_questions': sum(r['overflows_8192'] for r in rows),
                  'answer8192_reference_and_envelope_fit': answer_ok,
                  'final_C_not_yet_available': True, 'strict_final_C_preflight_before_answering': True,
                  'evaluation_output_records_observed': 0, 'inference_HTTP_sent': 0,
                  'CPU_endpoint_calls': counter.http_calls, 'CPU_seconds': time.monotonic()-started, 'rows': rows}
        lm.write(LOCAL/'rerank-jobs.json', jobs); lm.write(LOCAL/'C-reference-inputs.json', C_reference)
        lm.write(LOCAL/'token-counts.json', counter.counts)
        report['private_jobs_sha256'] = lm.file_hash(LOCAL/'rerank-jobs.json')
        report['private_C_reference_sha256'] = lm.file_hash(LOCAL/'C-reference-inputs.json')
        path = lm.RESULTS/'longmemeval-g1-window-diagnosis.json'
        lm.write(path, report)
        lines = ['# G1 CPU入力長診断（出力0件）', '',
          f'規則d2e7f4fを先にcommit。最大入力+120+128に10%余裕、必要{required}、選択{chosen}。', '',
          '|質問ID|候補数|rerank入力|最大候補|単独超過候補番号|初めて超過する累積候補番号|C dense参考入力|C予算上界見積＋出力余裕|',
          '|---|---:|---:|---:|---|---|---:|---:|']
        for r in rows:
            lines.append(f'|{r["id"]}|{r["candidate_count"]}|{r["rerank_input_tokens"]}|{r["max_candidate_tokens"]}|'
                         f'{r["single_candidate_overflow_numbers"]}|{r["first_prefix_crossing_candidate"]}|'
                         f'{r["C_dense_reference_input_tokens"]}|{r["C_envelope_plus_output_margin_estimate"]}|')
        lines += ['', '候補別unit/原発言ID/token数はJSON rows.candidates。本文は非公開。',
                  'C dense参考と構造予算見積は、rerank後の実入力の測定ではない。全30の実入力を回答生成前に再検査する。',
                  '旧8k並べ替え設計がこの長発話標本で成立しなかったのは運用上の一般化の限界。精度とは別に報告する。']
        (lm.RESULTS/'longmemeval-g1-window-diagnosis.md').write_text('\n'.join(lines)+'\n')
        print({k: report[k] for k in ('rerank_distribution', 'required_with10percent', 'selected_rerank_context',
                                     'overflow8k_questions', 'answer8192_reference_and_envelope_fit')})
    finally:
        lm.write(LOCAL/'token-counts.json', counter.counts)


if __name__ == '__main__':
    diagnose()
