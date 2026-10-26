# Real demo transcript: Gemma 4 E2B on llama-server, with and without the tsuzuki proxy (temperature 0).
import json, sys, time, urllib.request
lang, token, out = sys.argv[1], sys.argv[2], sys.argv[3]
PROXY, MODEL, SIDECAR = "http://127.0.0.1:8395/v1", "http://127.0.0.1:8382/v1", "http://127.0.0.1:8394"
SYSTEM = {"en": "You are a helpful assistant. Answer in one or two short sentences.",
          "ja": "あなたは親切なアシスタントです。1〜2文で短く答えてください。"}[lang]
FIRST = {"en": "Hi! I'm a second-year student, and I just started a part-time job at a ramen shop called Menya Kaze.",
         "ja": "来週の金曜日、大阪で友達の結婚式があって、スピーチを頼まれてるんだ。"}[lang]
LATER = {"en": "Where do I work part-time again?", "ja": "結婚式ってどこであるんだっけ？"}[lang]

def post(url, body, auth=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json",
                                 **({"authorization": f"Bearer {auth}"} if auth else {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

def chat(url, user, auth=None):
    t = time.time()
    r = post(f"{url}/chat/completions", {"model": "local-e2b", "temperature": 0, "seed": 0,
             "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}, auth)
    return r["choices"][0]["message"]["content"], round(time.time() - t, 2)

record = {"lang": lang, "system": SYSTEM, "model": "Gemma 4 E2B (gemma-4-E2B_q4_0-it.gguf) on llama-server b10964, temperature 0, seed 0"}
record["chat1"] = {"user": FIRST}
record["chat1"]["assistant"], record["chat1"]["seconds"] = chat(PROXY, FIRST, token)
for _ in range(100):  # the windows are embedded just after the reply
    with urllib.request.urlopen(urllib.request.Request(f"{SIDECAR}/health", headers={"authorization": f"Bearer {token}"})) as r:
        if json.load(r)["stats"]["unindexed"] == 0:
            break
    time.sleep(0.1)
record["recall"] = post(f"{SIDECAR}/v1/memory/recall", {"query": LATER, "session": "a-new-chat"}, token)["text"]
record["chat2"] = {"user": LATER}
record["chat2"]["assistant"], record["chat2"]["seconds"] = chat(PROXY, LATER, token)
record["without"] = {"user": LATER}
record["without"]["assistant"], record["without"]["seconds"] = chat(MODEL, LATER)
json.dump(record, open(out, "w"), ensure_ascii=False, indent=2)
print(json.dumps(record, ensure_ascii=False, indent=2))
