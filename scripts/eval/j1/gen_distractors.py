#!/usr/bin/env python3
"""J1 distractor sessions: generic Japanese user-assistant chats written by Gemma 4 E2B (no personal facts).

    python3 gen_distractors.py --port 8093 --n 300 --out build/eval/memory/j1/distractors.json
The user only asks for information or advice; never mentions family, pets, home, job, health, dates, money or
other facts about themselves, so distractors cannot contradict the planted evidence. Seeded topic plan; resumable.
"""
from __future__ import annotations
import argparse, json, random, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOPICS = [
 '和食の煮物の基本', 'パスタのソースの種類', 'パンの発酵の仕組み', 'コーヒーの淹れ方', '紅茶の種類', '日本酒の分類', 'スパイスカレーの作り方',
 '冷凍保存のコツ', 'お弁当の詰め方', '発酵食品の歴史', '洗濯表示の読み方', '掃除機の選び方', '観葉植物の育て方', '家庭菜園の土づくり',
 '本棚の整理術', '騒音を減らす工夫', '電気代を抑える家電の使い方', 'ノートパソコンの選び方', 'スマホのバッテリーを長持ちさせる方法',
 'パスワード管理の考え方', 'Wi-Fiが遅い原因', 'プログラミング言語の比較', '表計算ソフトの関数', 'プレゼン資料の作り方', 'メールの書き方',
 '敬語の使い分け', '英単語の覚え方', '読書感想文の書き方', '日本史の鎌倉時代', '世界史の産業革命', '宇宙の成り立ち', '光合成の仕組み',
 '天気予報の読み方', '台風の仕組み', '地震のメカニズム', '恐竜の絶滅', '深海の生き物', '渡り鳥の不思議', '猫の習性一般', '犬の種類一般',
 '京都の観光地', '北海道の名物', '沖縄の文化', '温泉の泉質', '登山の装備', 'キャンプの基本', '自転車のメンテナンス一般', 'ランニングフォーム',
 'ストレッチの方法', '睡眠の質を上げる一般的な工夫', '水分補給の目安', 'チェスのルール', '将棋の戦法', '囲碁の基本', '写真の構図', '水彩画の技法',
 'ギターのコード', 'クラシック音楽の時代区分', 'ジャズの歴史', '映画の撮影技法', 'アニメの制作工程', '小説のプロット作り', '俳句の作り方',
 '折り紙の折り方', '手帳の使い方一般', '時間管理の手法', 'ポモドーロ・テクニック', '集中力を高める環境', '確定申告の一般知識', '投資信託の仕組み一般',
 '保険の種類一般', '引っ越しの一般的な段取り', '粗大ごみの出し方一般', '防災グッズ', '応急手当の基本', '虫よけの方法', '靴の手入れ', 'シャツのアイロン',
 '料理の計量の基本', '包丁の研ぎ方', '鉄のフライパンの手入れ', '炊飯器のお手入れ', '電子レンジの仕組み', '冷蔵庫の収納', 'お茶の歴史', '和菓子の種類',
]
SCHEMA = {'type': 'object', 'properties': {'turns': {'type': 'array', 'minItems': 8, 'maxItems': 12, 'items': {'type': 'object',
          'properties': {'role': {'type': 'string', 'enum': ['user', 'assistant']}, 'content': {'type': 'string'}}, 'required': ['role', 'content']}}},
          'required': ['turns']}
PROMPT = ('次の話題について、ユーザーとAIアシスタントの自然な日本語のチャットを作ってください。\n話題: {topic}\n切り口: {angle}\n'
          'ルール:\n- 8〜12発言。userとassistantが交互に話し、userから始める。\n'
          '- userは情報や助言を求めるだけ。自分・家族・ペット・住まい・仕事・学校・健康・予定・日付・金額・持ち物など、'
          '自分自身についての事実は一切話さない。\n- assistantは丁寧に、1回150〜300字で答える。\n- 固有の人名は出さない。\n'
          'JSONで {{"turns": [{{"role": "user", "content": "..."}}, ...]}} の形だけを出力する。')
ANGLES = ['初心者向けに', 'よくある失敗', '歴史的な背景', '比較しながら', '手順を追って', '意外な豆知識', 'Q&A形式で', '道具の選び方']


def call(port, prompt):
    body = {'model': 'e2b', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.8, 'top_p': 0.95, 'max_tokens': 3000,
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'chat', 'schema': SCHEMA}}}
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(), headers={'content-type': 'application/json'})
    out = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return json.loads(out['choices'][0]['message']['content'])['turns']


def valid(turns):
    if len(turns) < 8 or turns[0]['role'] != 'user':
        return False
    return all(t['role'] == ('user' if i % 2 == 0 else 'assistant') and t['content'].strip() for i, t in enumerate(turns))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8093)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    done = json.loads(out.read_text()) if out.exists() else {}
    rng = random.Random('j1-distractors-v1')
    plan = [(f'd{i:03d}', rng.choice(TOPICS), rng.choice(ANGLES)) for i in range(a.n)]
    todo = [p for p in plan if p[0] not in done]

    def one(p):
        sid, topic, angle = p
        for _ in range(3):
            try:
                turns = call(a.port, PROMPT.format(topic=topic, angle=angle))
                if valid(turns):
                    return sid, {'topic': topic, 'angle': angle, 'turns': turns}
            except Exception as e:  # retry
                print('retry', sid, e, file=sys.stderr)
        return sid, None
    with ThreadPoolExecutor(a.workers) as pool:
        for k, (sid, v) in enumerate(pool.map(one, todo), 1):
            if v:
                done[sid] = v
            if k % 10 == 0 or k == len(todo):
                out.write_text(json.dumps(done, ensure_ascii=False, indent=0))
                print(f'{len(done)}/{a.n}', flush=True)
    out.write_text(json.dumps(done, ensure_ascii=False, indent=0))


if __name__ == '__main__':
    main()
