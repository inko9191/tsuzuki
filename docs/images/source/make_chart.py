# README results chart: accuracy with tsuzuki's memory vs the whole history, per pre-registered test (numbers copied
# from the README's result tables, primary judge). Emphasis form: memory in the accent, full history in gray, every
# bar labeled with its value. One PNG per language and theme (light / dark, GitHub's page colors).
import os, sys, unicodedata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

def font_path(name):
    for f in os.listdir("/System/Library/Fonts"):
        if unicodedata.normalize("NFC", f) == name:
            return os.path.join("/System/Library/Fonts", f)
    raise SystemExit(name)

for f in ("ヒラギノ角ゴシック W3.ttc", "ヒラギノ角ゴシック W6.ttc"):
    font_manager.fontManager.addfont(font_path(f))
REG = font_manager.FontProperties(fname=font_path("ヒラギノ角ゴシック W3.ttc"))
BOLD = font_manager.FontProperties(fname=font_path("ヒラギノ角ゴシック W6.ttc"))

# (model, test, memory, full history, memory input tokens, full input tokens, difference, verdict key)
TESTS = [
    ("Gemma 4 E2B", "E1", "en", 0.525, 0.375, "2.9k", "117k", "+0.15", "equiv"),
    ("Gemma 4 E2B", "J1", "ja", 0.550, 0.275, "3.2k", "48k", "+0.28", "better"),
    ("Qwen 27B", "U2", "en", 0.675, 0.450, "2.9k", "116k", "+0.23", "better"),
    ("Qwen 27B", "J1-Q", "ja", 0.725, 0.750, "3.2k", "45k", "−0.03", "noise"),
]
TEXT = {
    "en": {"title": "Correct answers: memory vs. the whole history",
           "sub": "Pre-registered tests, 40 questions each, local models on llama.cpp",
           "mem": "tsuzuki: 2,048-token memory", "full": "the whole history in the context",
           "lang": {"en": "English", "ja": "Japanese"},
           "tokens": "{t} · input {m} vs {f} tokens",
           "verdict": {"equiv": "at least equivalent", "better": "better", "noise": "same within noise"},
           "foot": "English: LongMemEval-S. Japanese: a self-made test. 95% intervals in Results."},
    "ja": {"title": "正答率：記憶と全履歴",
           "sub": "事前登録した試験（各 40 問）、llama.cpp 上のローカルモデル",
           "mem": "tsuzuki の記憶（2,048 トークン）", "full": "全履歴をそのまま渡す",
           "lang": {"en": "英語", "ja": "日本語"},
           "tokens": "{t} · 入力 {m} 対 {f} トークン",
           "verdict": {"equiv": "同等以上", "better": "上回る", "noise": "誤差の範囲"},
           "foot": "英語は LongMemEval-S、日本語は自作の試験。95% 区間は「結果」に。"},
}
THEME = {
    "light": {"bg": "#ffffff", "ink": "#1f2328", "muted": "#59636e", "grid": "#e6e8eb", "mem": "#2a78d6", "full": "#8c8a83"},
    "dark": {"bg": "#0d1117", "ink": "#e6edf3", "muted": "#9198a1", "grid": "#262c36", "mem": "#3987e5", "full": "#6f6e69"},
}

def bar(ax, y, value, h, color):
    """A bar from 0 to `value`, square at the baseline, rounded at the data end."""
    r = 0.012
    ax.add_patch(Rectangle((0, y), max(value - r, 0), h, facecolor=color, edgecolor="none"))
    ax.add_patch(FancyBboxPatch((max(value - 2 * r, 0), y), 2 * r, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=color, edgecolor="none", mutation_aspect=h / (2 * r) * 0.5))

def chart(lang, theme, out):
    T, C = TEXT[lang], THEME[theme]
    fig = plt.figure(figsize=(6.4, 6.6), dpi=200, facecolor=C["bg"])
    ax = fig.add_axes((0.05, 0.07, 0.80, 0.76))
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 1)
    # Every bar carries its value, so there are no gridlines; each pair has its own baseline.
    slot, bar_h, gap = 1.3, 0.24, 0.03
    n = len(TESTS)
    ax.set_ylim(slot - 1.0, n * slot + 0.02)
    for i, (model, test, tl, mem, full, mt, ft, diff, verdict) in enumerate(TESTS):
        top = (n - i) * slot
        ax.text(0, top - 0.04, f"{model} · {T['lang'][tl]}", fontproperties=BOLD, fontsize=11.5, color=C["ink"], va="top")
        ax.text(1.0, top - 0.04, f"{diff}  {T['verdict'][verdict]}", fontproperties=BOLD if verdict == "better" else REG,
                fontsize=10.5, color=C["ink"] if verdict == "better" else C["muted"], va="top", ha="right")
        ax.text(0, top - 0.27, T["tokens"].format(t=test, m=mt, f=ft), fontproperties=REG, fontsize=9, color=C["muted"], va="top")
        y_mem = top - 0.47 - bar_h
        y_full = y_mem - gap - bar_h
        ax.plot([0, 0], [y_full - 0.03, y_mem + bar_h + 0.03], color=C["muted"], lw=1.0, zorder=3, solid_capstyle="butt")
        for y, v, color in ((y_mem, mem, C["mem"]), (y_full, full, C["full"])):
            bar(ax, y, v, bar_h, color)
            ax.text(v + 0.015, y + bar_h / 2, f"{v:.3f}", fontproperties=REG, fontsize=10, color=C["ink"], va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.05, 0.965, T["title"], fontproperties=BOLD, fontsize=14.5, color=C["ink"], va="top")
    fig.text(0.05, 0.925, T["sub"], fontproperties=REG, fontsize=9.5, color=C["muted"], va="top")
    lx = 0.05
    for key, color in (("mem", C["mem"]), ("full", C["full"])):
        fig.patches.append(Rectangle((lx, 0.874), 0.022, 0.016, transform=fig.transFigure, facecolor=color, edgecolor="none"))
        t = fig.text(lx + 0.03, 0.882, T[key], fontproperties=REG, fontsize=9.5, color=C["ink"], va="center")
        fig.canvas.draw()
        lx = fig.transFigure.inverted().transform(t.get_window_extent().corners()[-1])[0] + 0.04
    fig.text(0.05, 0.025, T["foot"], fontproperties=REG, fontsize=8.5, color=C["muted"], va="bottom")
    fig.savefig(out, facecolor=C["bg"])
    plt.close(fig)

for lang in ("en", "ja"):
    for theme in ("light", "dark"):
        out = os.path.join(sys.argv[1], f"results-{lang}-{theme}.png")
        chart(lang, theme, out)
        print(out, os.path.getsize(out))
