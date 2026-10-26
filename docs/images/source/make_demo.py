# Animated README demo from a real transcript (transcript-<lang>.json): chat 1, a new chat that recalls it, and the
# same model without tsuzuki. Every message text is the captured output, unchanged.
import json, os, sys, unicodedata
from PIL import Image, ImageDraw, ImageFont

lang, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
t = json.load(open(src))

def font_path(name):
    for f in os.listdir("/System/Library/Fonts"):
        if unicodedata.normalize("NFC", f) == name:
            return os.path.join("/System/Library/Fonts", f)
    raise SystemExit(f"font not found: {name}")

REG, BOLD = font_path("ヒラギノ角ゴシック W3.ttc"), font_path("ヒラギノ角ゴシック W6.ttc")
W, PAD, GAP = 600, 24, 14
F_BODY, F_SMALL, F_LABEL = ImageFont.truetype(REG, 23), ImageFont.truetype(REG, 16), ImageFont.truetype(BOLD, 16)
INK, MUTED, LINE = (31, 35, 40), (87, 96, 106), (216, 222, 228)
BG, PANEL, USER, ASSIST = (244, 246, 248), (255, 255, 255), (42, 120, 214), (238, 241, 244)
MEM_BG, MEM_INK = (232, 241, 252), (29, 79, 145)

L = {
    "en": {"chat1": "Chat 1", "new": "New chat", "recalled": "tsuzuki recalled, from 2026/10/26:",
           "without": "The same model without tsuzuki:", "model": "Gemma 4 E2B · llama.cpp",
           "foot": ["Real output, temperature 0 · all on one Mac", "Gemma 4 E2B (3.3 GB) + tsuzuki's proxy"]},
    "ja": {"chat1": "チャット 1", "new": "新しいチャット", "recalled": "tsuzuki が思い出した記憶（2026/10/26）：",
           "without": "同じモデルで tsuzuki なし：", "model": "Gemma 4 E2B · llama.cpp",
           "foot": ["実際の出力（温度 0）· 全部 1 台の Mac の中", "Gemma 4 E2B（3.3 GB）＋ tsuzuki のプロキシ"]},
}[lang]

measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))

def wrap(text, font, width):
    """Lines of `text` within `width` px: by words for English, by characters for Japanese (no line starts with 、。)."""
    lines, line = [], ""
    tokens = text.split(" ") if lang == "en" else list(text)
    for tok in tokens:
        cand = (line + " " + tok).strip() if lang == "en" else line + tok
        if measure.textlength(cand, font=font) <= width or not line or (lang == "ja" and tok in "、。」）！？"):
            line = cand
        else:
            lines.append(line)
            line = tok
    if line:
        lines.append(line)
    return lines

def line_h(font):
    return int(font.size * 1.45)

# Blocks, top to bottom: (kind, payload); each knows its height.
recalled = next(l for l in t["recall"].splitlines() if "] user: " in l).split("] user: ", 1)[1]
BUBBLE_W = int((W - 2 * PAD) * 0.8)
blocks = [
    ("header", L["chat1"]),
    ("user", t["chat1"]["user"]),
    ("assistant", t["chat1"]["assistant"]),
    ("divider", L["new"]),
    ("user", t["chat2"]["user"]),
    ("memory", recalled),
    ("assistant", t["chat2"]["assistant"]),
    ("without", t["without"]["assistant"]),
]

def height(kind, text):
    if kind == "header":
        return line_h(F_LABEL) + 6
    if kind == "divider":
        return line_h(F_LABEL) + 14
    if kind in ("user", "assistant"):
        return len(wrap(text, F_BODY, BUBBLE_W - 32)) * line_h(F_BODY) + 24
    inner = W - 2 * PAD - 32
    return line_h(F_LABEL) + len(wrap(text, F_BODY, inner)) * line_h(F_BODY) + 30

foot_lines = L["foot"]
H = PAD + sum(height(k, x) + GAP for k, x in blocks) + len(foot_lines) * line_h(F_SMALL) + PAD + 6

def draw_text(d, xy, lines, font, fill):
    x, y = xy
    for ln in lines:
        d.text((x, y), ln, font=font, fill=fill)
        y += line_h(font)

def frame(shown):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((8, 8, W - 9, H - 9), radius=18, fill=PANEL, outline=LINE, width=2)
    y = PAD
    for kind, text in blocks[:shown]:
        h = height(kind, text)
        if kind == "header":
            d.text((PAD, y), text, font=F_LABEL, fill=MUTED)
            mw = d.textlength(L["model"], font=F_SMALL)
            d.text((W - PAD - mw, y + 1), L["model"], font=F_SMALL, fill=MUTED)
        elif kind == "divider":
            mid = y + h // 2
            tw = d.textlength(text, font=F_LABEL)
            d.line((PAD, mid, W / 2 - tw / 2 - 12, mid), fill=LINE, width=2)
            d.line((W / 2 + tw / 2 + 12, mid, W - PAD, mid), fill=LINE, width=2)
            d.text((W / 2 - tw / 2, mid - line_h(F_LABEL) // 2 + 2), text, font=F_LABEL, fill=MUTED)
        elif kind in ("user", "assistant"):
            lines = wrap(text, F_BODY, BUBBLE_W - 32)
            bw = max(d.textlength(ln, font=F_BODY) for ln in lines) + 32
            x0 = W - PAD - bw if kind == "user" else PAD
            d.rounded_rectangle((x0, y, x0 + bw, y + h), radius=16, fill=USER if kind == "user" else ASSIST)
            draw_text(d, (x0 + 16, y + 11), lines, F_BODY, (255, 255, 255) if kind == "user" else INK)
        else:
            memory = kind == "memory"
            d.rounded_rectangle((PAD, y, W - PAD, y + h), radius=12, fill=MEM_BG if memory else BG,
                                outline=USER if memory else LINE, width=2)
            d.text((PAD + 16, y + 10), L["recalled"] if memory else L["without"], font=F_LABEL, fill=MEM_INK if memory else MUTED)
            body = f"“{text}”" if memory and lang == "en" else (f"「{text}」" if memory else text)
            draw_text(d, (PAD + 16, y + 12 + line_h(F_LABEL)), wrap(body, F_BODY, W - 2 * PAD - 32), F_BODY,
                      MEM_INK if memory else MUTED)
        y += h + GAP
    draw_text(d, (PAD, H - PAD - len(foot_lines) * line_h(F_SMALL) + 2), foot_lines, F_SMALL, MUTED)
    return img

# How long each step stays (ms) before the next block appears; the last one is the final hold.
HOLD = [700, 1500, 1900, 900, 1300, 1700, 2000, 5200]
frames = [frame(n) for n in range(1, len(blocks) + 1)]
frames[0].save(out, save_all=True, append_images=frames[1:], duration=HOLD, loop=0, optimize=True, disposal=1)
frames[-1].save(out.replace(".gif", "-still.png"))
print(out, W, H, os.path.getsize(out))
