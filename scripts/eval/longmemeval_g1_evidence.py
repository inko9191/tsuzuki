"""Exploratory saved-input evidence coverage. CPU/files only; no HTTP or model API."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import locomo_answer_methods as answers

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'build/eval/memory/longmemeval-g1'
LOCAL = BASE/'continuation'
PRIVATE = LOCAL/'exploratory-evidence'
PUBLIC = ROOT/'results/eval/memory'
STRATA = ('multi-session','temporal-reasoning','knowledge-update','single-session','abstention')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def coverage(question, turn_ids):
    qid = question['question_id']
    session_ids = question['haystack_session_ids']
    evidence = set(question['answer_session_ids'])
    if not evidence or not evidence <= set(session_ids):
        raise ValueError('missing/unmapped evidence metadata')
    selected, indices = set(turn_ids), set()
    for tid in selected:
        match = re.fullmatch(re.escape(qid)+r'/D(\d+):(\d+)',tid)
        if not match:
            raise ValueError('unrecognized final-input turn ID')
        index, position = (int(x)-1 for x in match.groups())
        if not 0 <= index < len(session_ids) or not 0 <= position < len(question['haystack_sessions'][index]):
            raise ValueError('out-of-range final-input turn')
        indices.add(index)
    represented = {session_ids[i] for i in indices}
    marked = []
    for i,(sid,session) in enumerate(zip(session_ids,question['haystack_sessions']),1):
        if sid in evidence:
            marked += [f'{qid}/D{i}:{j}' for j,m in enumerate(session,1) if m.get('has_answer') is True]
    return {'evidence_session_ids':sorted(evidence),'represented_evidence_ids':sorted(evidence&represented),
            'missing_evidence_ids':sorted(evidence-represented),'all_evidence_sessions':evidence <= represented,
            'any_evidence_session':bool(evidence&represented),'evidence_session_count':len(evidence),
            'represented_evidence_count':len(evidence&represented),'selected_original_session_ids':sorted(represented),
            'has_answer_marked_turns':marked,'selected_marked_turns':sorted(set(marked)&selected),
            'all_marked_turns_selected':set(marked)<=selected if marked else None,
            'duplicate_haystack_session_ids':sorted(s for s,n in Counter(session_ids).items() if n>1)}


def quadrant(present, correct):
    return ('present' if present else 'missing')+'_'+('correct' if correct else 'incorrect')


def summarize(rows, field='all_evidence_sessions'):
    counts = Counter(quadrant(r[field],r['C_correct']) for r in rows)
    return {k:counts[k] for k in ('present_correct','present_incorrect','missing_correct','missing_incorrect')}


def main():
    prepared = read(BASE/'run/prepared.json')
    plan = read(PUBLIC/'longmemeval-g1-window-deviation.json')
    verification = read(PUBLIC/'longmemeval-g1-verification.json')
    originals = {**plan['original_source_sha256'],**plan['source_sha256']}
    if sha(BASE/'run/prepared.json') != originals[str((BASE/'run/prepared.json').relative_to(ROOT))]:
        raise RuntimeError('protected prepared source changed')
    if sha(LOCAL/'received.jsonl') != verification['new_received_journal_sha256']:
        raise RuntimeError('measured output journal changed')
    if sha(PUBLIC/'longmemeval-g1-report.json') != verification['report_sha256']:
        raise RuntimeError('registered report changed')
    preserved = read(PUBLIC/'longmemeval-g1-finalization-deviation.json')
    for name, field in [('scored-rows.json','before_scored_rows_sha256'),('review.json','before_review_sha256')]:
        if sha(LOCAL/name) != preserved[field]:
            raise RuntimeError('registered labels/review changed: '+name)
    inputs = read(LOCAL/'answer-inputs.json')['C-final-norw']
    review = {r['id']:r for r in read(LOCAL/'review.json')}
    scores = {r['id']:r for r in read(LOCAL/'scored-rows.json')['C-final-norw']}
    registered = read(PUBLIC/'longmemeval-g1-report.json')
    tags = read(PRIVATE/'manual-error-tags.json')['tags']
    replies = {r['key']:r['reply'] for r in map(json.loads,(LOCAL/'received.jsonl').read_text().splitlines())}
    server = next(json.loads(s)['reply']['server'] for s in (LOCAL/'server-starts.jsonl').read_text().splitlines()
                  if json.loads(s)['reply']['label']=='C-final-norw')
    rows, audit = [], []
    for qid in plan['question_ids']:
        q, item = prepared['questions'][qid], inputs[qid]
        lookup = {t['id']:t for t in prepared['turns'][qid]}
        recent = [lookup[t] for t in item['recent_ids']]; memory = [lookup[t] for t in item['memory_ids']]
        messages = answers.messages(answers.render(recent),answers.render(memory),q['question'])
        if messages != item['messages']:
            raise RuntimeError('saved final-input provenance differs from rendered prompt')
        body = {'model':server['model_id'],'messages':messages,**answers.GENERATION,
                'response_format':{'type':'json_schema','json_schema':{'name':'answer','schema':answers.SCHEMA}}}
        reply = replies['answer|C-final-norw|'+qid]
        if digest({'body':body,'model':server['model_sha256']}) != reply['digest']:
            raise RuntimeError('final-input hash differs from actual sent request')
        row = coverage(q,item['recent_ids']+item['memory_ids'])
        retrieved_only = coverage(q,item['memory_ids'])
        score, reviewed = scores[qid], review[qid]['conditions']['C-final-norw']
        if score['correct'] != reviewed['judge']['label']:
            raise RuntimeError('registered correctness sources disagree')
        correct = bool(score['correct']); present = row['all_evidence_sessions']
        row.update(id=qid,stratum=score['stratum'],type=q['question_type'],C_correct=correct,
            C_abstention=bool(score['abstention']),quadrant=quadrant(present,correct),
            attribution='none' if present and correct else 'answer_side_ID_proxy' if present else 'retrieval_side_noncoverage',
            observed_error_category=tags[qid]['category'] if not correct else None,
            retrieved_only_all=retrieved_only['all_evidence_sessions'],retrieved_only_any=retrieved_only['any_evidence_session'],
            retrieved_only_evidence_count=retrieved_only['represented_evidence_count'],
            retrieved_only_missing_evidence_ids=retrieved_only['missing_evidence_ids'],
            final_recent_turn_ids=item['recent_ids'],final_retrieved_turn_ids=item['memory_ids'],sent_request_digest_verified=True)
        audit.append({**row,'question':q['question'],'gold':q['answer'],'C_answer':reviewed['answer'],
            'manual_error_tag':tags.get(qid),'selected_turns':recent+memory,
            'marked_turn_texts':[lookup[t] for t in row['has_answer_marked_turns']]})
        # Annotation text and source messages stay private; public rows retain IDs/counts only.
        rows.append(row)
    if len(rows)!=30 or sum(r['C_correct'] for r in rows)!=15 or set(scores)!=set(plan['question_ids']):
        raise RuntimeError('registered full30 cohort changed')
    result = {'time':datetime.now().astimezone().isoformat(),'analysis':'exploratory post-outcome evidence-session coverage',
        'questions':30,'language':'English only','new_HTTP_requests':0,
        'rule':'Primary present = EVERY distinct answer_session_id represented by at least one original turn in actual final C prompt. Partial coverage is missing. Supplemental any-ID and retrieved-only retained.',
        'final_input_scope':'recent conversation1024 plus retrieved memory512; original session IDs derived from D index provenance, not literal prompt-ID strings',
        'correctness':'unchanged registered Qwen labels, not new scoring; same model answers and judges',
        'limits':['ID coverage does not prove relevant turn or sufficient evidence was included; marked-turn coverage is additional annotation proxy, not semantic sufficiency.',
            'Missing evidence IDs can coexist with correct answers (updated values or unanswerable questions); retrieval-side noncoverage includes correct answers by requested accounting rule.',
            'Answer-side label is requested ID-level attribution, not causal proof of generator failure. Annotation gaps/assembly loss or judge disagreement can contribute.',
            'Fixed30 stratified English sample; self-judge bias/training contamination/history dependence unknown; descriptive exploratory counts only.',
            'No design/criteria changes from these results; require confirmation on a new sample before using as design-change evidence. No new sample/experiment authorized here.'],
        'quadrants_all30':summarize(rows),'quadrants_answerable24':summarize([r for r in rows if r['stratum']!='abstention']),
        'quadrants_any_ID_supplemental':summarize(rows,'any_evidence_session'),
        'quadrants_retrieved_memory_only_supplemental':summarize(rows,'retrieved_only_all'),
        'by_stratum':{s:summarize([r for r in rows if r['stratum']==s]) for s in STRATA},
        'retrieval_side_noncoverage_all30':sum(not r['all_evidence_sessions'] for r in rows),
        'answer_side_failures_ID_proxy':sum(r['all_evidence_sessions'] and not r['C_correct'] for r in rows),
        'incorrect_with_missing_sessions':sum(not r['all_evidence_sessions'] and not r['C_correct'] for r in rows),
        'answer_side_failure_forms':{k:sum(r['all_evidence_sessions'] and not r['C_correct'] and r['observed_error_category']==k for r in rows)
            for k in ('old_value','temporal_order','abstention','other')},
        'answer_side_forms_by_stratum':{s:{k:sum(r['stratum']==s and r['all_evidence_sessions'] and not r['C_correct'] and r['observed_error_category']==k for r in rows)
            for k in ('old_value','temporal_order','abstention','other')} for s in STRATA},
        'ID_present_incorrect_with_marked_turns_missing':sum(r['all_evidence_sessions'] and not r['C_correct'] and r['all_marked_turns_selected'] is False for r in rows),
        'temporal_order_error_missing_ID_question_ids':[r['id'] for r in rows if not r['all_evidence_sessions'] and r['observed_error_category']=='temporal_order'],
        'rows':rows,'registered_main_result_unchanged':registered['C_minus_A_semantic_primary']['overall'],
        'swap_and_time_scope':'saved original start swap8.99GiB; original timing reference only; no GPU/HTTP run in this CPU analysis'}
    PRIVATE.mkdir(exist_ok=True);write(PRIVATE/'evidence-audit.json',audit)
    write(PUBLIC/'longmemeval-g1-evidence-exploratory.json',result)
    lines=['# G1 根拠セッション被覆（探索的・英語30問）','',
        '新しいHTTPなし。標本・設計・正誤ラベル・採点基準は変更していない。新しい標本で確認するまで設計変更の根拠にしない。',
        '', '根拠あり＝answer_session_idsの全ID由来の発言が最終C入力に1発言以上ある。一部欠落は「根拠なし（全件未被覆）」とする。最終入力は最近履歴＋並べ替え後の検索記憶。IDの由来を保存入力と実HTTP request digestに全30問で照合。',
        '', '|登録層|根拠あり・正解|根拠あり・不正解/棄権|根拠なし・正解|根拠なし・不正解/棄権|',
        '|---|---:|---:|---:|---:|']
    for s,counts in [('全30',result['quadrants_all30']),('回答可能24',result['quadrants_answerable24']),*result['by_stratum'].items()]:
        lines.append('|'+s+'|'+'|'.join(str(n) for n in counts.values())+'|')
    lines += ['',f'依頼のID基準による回答側失敗 {result["answer_side_failures_ID_proxy"]}問、検索側未被覆 {result["retrieval_side_noncoverage_all30"]}問。後者は正解も含み、実際のC不正解15問の分解では回答側11＋検索側4。棄権問題は根拠セッションが欠けても正しく棄権できる。',
        '', '最終入力の範囲を限定した補足：最近履歴を除き「検索記憶512token部分だけ」を見ると、次のとおり。主集計の定義は変更しない。',
        '', '|入力範囲|全根拠あり・正解|全根拠あり・不正解|全根拠未被覆・正解|全根拠未被覆・不正解|',
        '|---|---:|---:|---:|---:|',
        '|検索記憶のみ|'+'|'.join(str(n) for n in result['quadrants_retrieved_memory_only_supplemental'].values())+'|',
        '', '|回答側11問の観測形|件数|','|---|---:|']
    for k,label in [('old_value','古い値を答えた'),('temporal_order','時間順の誤り'),('abstention','棄権'),('other','その他（比較/採点境界）')]:
        lines.append(f'|{label}|{result["answer_side_failure_forms"][k]}|')
    lines += ['', '|登録層|古い値|時間順|棄権|その他|','|---|---:|---:|---:|---:|']
    for s,counts in result['answer_side_forms_by_stratum'].items():lines.append('|'+s+'|'+'|'.join(str(n) for n in counts.values())+'|')
    lines += ['',f'根拠IDが全件ある不正解11問のうち{result["ID_present_incorrect_with_marked_turns_missing"]}問では、has_answer=true注釈の発言が少なくとも1つ欠落。セッションID被覆を根拠内容の完備と解釈しない。古い値4問のうち1問では更新発言が最終入力にない。時間順誤りは別に1問あるが根拠ID欠落のため検索側へ数えた。その他の1問には保存済みCodexとQwenの採点不一致があり、登録Qwen不正解を保持。',
        '', '補助の「少なくとも1根拠IDあり」では全30問が該当（正解15・不正解15、未被覆0）。全件被覆21問と同じ意味ではない。検索記憶だけの被覆もJSON各問行へ保存。因果帰属や種類別有意性は主張しない。',
        '', '|問ID|登録層|被覆ID/根拠ID|全件被覆|C登録正誤|4区分|ID基準の帰属|観測誤答形|',
        '|---|---|---:|---|---|---|---|---|']
    for r in rows:
        lines.append(f'|{r["id"]}|{r["stratum"]}|{r["represented_evidence_count"]}/{r["evidence_session_count"]}|'
            f'{"あり" if r["all_evidence_sessions"] else "一部欠落"}|{"正解" if r["C_correct"] else "棄権" if r["C_abstention"] else "不正解"}|'
            f'{r["quadrant"]}|{r["attribution"]}|{r["observed_error_category"] or "—"}|')
    lines += ['', '根拠/欠落セッションID・選択発言IDは公開JSON。質問・正解・本文・手作業分類根拠はprivate evidence-audit.jsonのみ。',
        '正誤は同じQwenによる自己採点。英語30問/層別少数・学習混入/履歴依存は未知。開始swap約8.99GiBで元時間は参考値。この分析はCPUのみで新しい性能測定ではない。']
    (PUBLIC/'longmemeval-g1-evidence-exploratory.md').write_text('\n'.join(lines)+'\n')
    files=[BASE/'run/prepared.json',LOCAL/'answer-inputs.json',LOCAL/'scored-rows.json',LOCAL/'review.json',LOCAL/'received.jsonl',
           LOCAL/'server-starts.jsonl',PUBLIC/'longmemeval-g1-report.json',PRIVATE/'manual-error-tags.json']
    write(PUBLIC/'longmemeval-g1-evidence-verification.json',{'time':result['time'],'questions':30,'new_HTTP_requests':0,
        'all_final_prompt_reconstructions_and_actual_request_digests_match':True,'original_prepared_hash_match':True,
        'unchanged_registered_15_correct_and_main_comparison':True,'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},
        'private_audit_sha256':sha(PRIVATE/'evidence-audit.json'),'analysis_source_sha256':sha(Path(__file__)),
        'test_source_sha256':sha(ROOT/'scripts/eval/tests/test_longmemeval_g1_evidence.py'),
        'result_sha256':sha(PUBLIC/'longmemeval-g1-evidence-exploratory.json'),
        'markdown_sha256':sha(PUBLIC/'longmemeval-g1-evidence-exploratory.md')})
    print(json.dumps({'quadrants':result['quadrants_all30'],'answer_side_forms':result['answer_side_failure_forms'],
                      'marked_turn_missing':result['ID_present_incorrect_with_marked_turns_missing']},ensure_ascii=False))


if __name__=='__main__':
    main()
