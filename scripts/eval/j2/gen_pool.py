#!/usr/bin/env python3
"""J2 pools (docs/MEMORY_MISFIRE_M2_PREREG_DRAFT.md), written by Gemma 4 E2B: generic Japanese user-assistant chats
for the histories (as in J1, on 100 new topics), and, separately, first messages to an AI assistant for the probes.

    python3 gen_pool.py sessions --port 8094 --n 300 --out build/eval/memory/j2/noise.json
    python3 gen_pool.py probes --port 8094 --n 400 --out build/eval/memory/j2/probes.json
The user never says anything about themself, so neither pool can contradict the written evidence. Seeded; resumable.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'j1'))
import gen_distractors as j1  # noqa: E402

TOPICS = [
 # computers and software
 'Pythonのリスト内包表記', 'Gitのブランチの使い方', 'SQLのJOIN', '正規表現の基本', 'JavaScriptの非同期処理', 'CSSのレイアウト',
 'Linuxの基本コマンド', 'データベースの正規化', 'APIとは何か', '機械学習の過学習', 'クラウドストレージの仕組み', '二段階認証',
 '画像のファイル形式の違い', 'キーボードのショートカット', '動画編集ソフトの基本',
 # language and writing
 '小論文の構成', '履歴書の書き方一般', '英語の時制', 'ハングルの読み方', '中国語の声調', '手紙の時候の挨拶', '要約のコツ',
 'ことわざの由来', '四字熟語', '句読点の使い方',
 # science
 'ブラックホール', '量子コンピュータ', 'DNAの仕組み', 'ワクチンの仕組み', '虹ができる理由', '月の満ち欠け', '火山の噴火',
 '磁石の仕組み', '電池の仕組み', '酸性とアルカリ性', '進化論', '元素の周期表',
 # history and society
 '江戸時代の暮らし', '戦国時代の武将一般', '古代ローマ', 'エジプトのピラミッド', '明治維新', '大航海時代', '選挙の仕組み',
 '裁判員制度', '国連の役割', '為替の仕組み',
 # mathematics
 '確率の考え方', '平均と中央値', '微分の意味', '素数の不思議', '図形の証明', '暗算のコツ',
 # hobbies and arts
 'ボードゲームの種類', '編み物の基本', '陶芸', '書道', 'カリグラフィー', '盆栽', '天体観測', '釣りの基本', '鉄道の楽しみ方',
 '切手収集', '落語', '歌舞伎',
 # music, film, books
 'ピアノの練習法', '作曲の基礎', '合唱の発声', '推理小説の歴史', '映画のジャンルの違い', '漫画の描き方',
 # sports
 '水泳のクロール', 'テニスのルール', 'サッカーのポジション', '野球のルール', 'ヨガの基本', '筋トレのフォーム一般',
 # home and everyday life
 '部屋のカビ対策', '結露対策', '窓掃除', '布団の干し方', '食器洗い機の使い方', '衣替え', '家計簿のつけ方一般',
 'ふるさと納税の仕組み', '年賀状のマナー', '冠婚葬祭のマナー',
 # food
 '寿司の歴史', 'ラーメンの種類', 'チーズの種類', 'ワインの基本', '味噌の種類', '中華料理の調味料',
 # nature
 '桜の品種', '紅葉の仕組み', 'クラゲの生態', 'ミツバチの社会', '熱帯雨林',
 # travel and geography
 '世界遺産', '時差の仕組み',
]
STYLES = ['短い一言の質問', '具体的なお願い（文章や例を作ってほしい）', '二つを比べてどちらがいいか聞く', '手順を聞く',
          '理由や仕組みを聞く', 'おすすめを聞く', '自分の理解が合っているか確かめる', 'アイデアを出してほしい']
TONES = ['くだけた話し言葉', '丁寧語']
PROBE_SCHEMA = {'type': 'object', 'properties': {'message': {'type': 'string'}}, 'required': ['message']}
PROBE_PROMPT = ('AIアシスタントに話しかける、利用者の最初の一言を日本語で1つ書いてください。\n話題: {topic}\n形: {style}\n口調: {tone}\n'
                'ルール:\n- 1〜4文、20〜300字。\n'
                '- 自分・家族・ペット・住まい・仕事・学校・健康・予定・日付・金額・持ち物など、自分自身についての事実は一切話さない。\n'
                '- 前の会話を前提にしない（「さっきの」「前に話した」などは使わない）。\n'
                'JSONで {{"message": "..."}} の形だけを出力する。')

assert len(TOPICS) == 100 and len(set(TOPICS)) == 100, len(TOPICS)
assert not set(TOPICS) & set(j1.TOPICS)


def call(port, prompt, schema, name, max_tokens):
    body = {'model': 'e2b', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.8, 'top_p': 0.95,
            'max_tokens': max_tokens, 'response_format': {'type': 'json_schema', 'json_schema': {'name': name, 'schema': schema}}}
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'content-type': 'application/json'})
    out = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return json.loads(out['choices'][0]['message']['content'])


def session(port, item):
    turns = call(port, j1.PROMPT.format(topic=item['topic'], angle=item['angle']), j1.SCHEMA, 'chat', 3000)['turns']
    return {**item, 'turns': turns} if j1.valid(turns) else None


def probe(port, item):
    text = call(port, PROBE_PROMPT.format(topic=item['topic'], style=item['style'], tone=item['tone']), PROBE_SCHEMA,
                'message', 400)['message'].strip()
    return {**item, 'text': text} if 20 <= len(text) <= 600 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['sessions', 'probes'])
    ap.add_argument('--port', type=int, default=8094)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text()) if out.exists() else {}
    rng = random.Random(f'j2-{a.kind}-v1')
    if a.kind == 'sessions':
        plan = [(f's{i:03d}', {'topic': rng.choice(TOPICS), 'angle': rng.choice(j1.ANGLES)}) for i in range(a.n)]
        make = session
    else:
        plan = [(f'p{i:03d}', {'topic': rng.choice(TOPICS), 'style': rng.choice(STYLES), 'tone': rng.choice(TONES)})
                for i in range(a.n)]
        make = probe
    todo = [p for p in plan if p[0] not in done]

    def one(p):
        key, item = p
        for _ in range(3):
            try:
                v = make(a.port, item)
                if v:
                    return key, v
            except Exception as e:  # retry
                print('retry', key, e, file=sys.stderr)
        return key, None
    with ThreadPoolExecutor(a.workers) as pool:
        for k, (key, v) in enumerate(pool.map(one, todo), 1):
            if v:
                done[key] = v
            if k % 10 == 0 or k == len(todo):
                out.write_text(json.dumps(done, ensure_ascii=False, indent=0))
                print(f'{a.kind} {len(done)}/{a.n}', flush=True)
    out.write_text(json.dumps(done, ensure_ascii=False, indent=0))


if __name__ == '__main__':
    main()
