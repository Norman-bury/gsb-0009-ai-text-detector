#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI 味识别器：判断一段中文是人写的还是 AI 写的。

纯 Python 标准库，全程离线。方法：
  1. 从 train.jsonl 统计出若干可解释特征在 human / ai 两类上的分布
     （句长波动、连接词密度、顿号密度、"其/该" 用字习惯、四字格密度等）。
  2. 再训练两个按类分开的字符 n-gram 语言模型，算文本的 n-gram 意外度。
  3. 分类器是高斯朴素贝叶斯 + n-gram 对数似然比，等价于一个手写线性模型，
     每个特征对最终分数的贡献可以逐项列出，因此判断过程完全可解释。

用法：
  python3 detector.py train                # 从 corpus/train.jsonl 统计特征，写 model.json
  python3 detector.py eval                 # 在 corpus/test.jsonl 上评估准确率
  python3 detector.py predict "一段中文"    # 判断一段文字，输出标签 + 逐项解释
  python3 detector.py predict -f file.txt  # 从文件读入文本
"""
import json
import math
import os
import re
import statistics
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE, "corpus", "train.jsonl")
TEST_PATH = os.path.join(BASE, "corpus", "test.jsonl")
MODEL_PATH = os.path.join(BASE, "model.json")

NGRAM_N = 3          # 字符 n-gram 的阶数
NGRAM_K = 0.1        # 加 k 平滑
MIN_SENTENCES = 3    # 文本至少这么多句，句长类特征才可信

# 连接词/套话候选表。具体哪些进入模型、权重多少，由训练语料统计决定，
# 这里只给候选集合（常见书面连接词与 AI 腔套话）。
# AI 腔高频动词/名词候选（是否入选、权重多少由训练语料统计决定）
STYLE_WORD_CANDIDATES = [
    "能够", "提供", "具有", "实现", "面临", "挑战", "不断", "显著",
    "这些", "这种", "这一", "机遇", "方面", "领域", "发展", "提升",
    "优化", "推动", "促进", "保障", "支撑", "赋能", "助力", "打造",
    "关键", "核心", "综合", "逐渐", "途径", "前景", "体系",
]

CONNECTIVE_CANDIDATES = [
    "然而", "因此", "此外", "同时", "值得注意的是", "值得注意", "综上所述", "综上",
    "首先", "其次", "再次", "最后", "不仅", "而且", "并且", "随着", "通过", "基于",
    "以及", "从而", "进而", "进一步", "总之", "总的来说", "具体而言", "具体来说",
    "例如", "另外", "为此", "与此同时", "另一方面", "一方面", "换句话说",
    "一般来说", "通常", "显著", "有效", "广泛", "重要",
]

SENT_SPLIT = re.compile(r"[。！？；!?;]+")
NON_CJK = re.compile(r"[^\u4e00-\u9fff]")


def split_sentences(text):
    return [s for s in SENT_SPLIT.split(text) if s.strip()]


def cjk_len(text):
    return len(NON_CJK.sub("", text))


def char_ngrams(text, n):
    chars = NON_CJK.sub("", text)
    return [chars[i:i + n] for i in range(len(chars) - n + 1)]


def four_char_chunks(text):
    """连续四个汉字且不含标点，视作一个四字格候选。"""
    chunks = []
    for m in re.finditer(r"[\u4e00-\u9fff]{4,}", text):
        run = m.group(0)
        chunks.extend(run[i:i + 4] for i in range(len(run) - 3))
    return chunks


# ---------------------------------------------------------------- 特征提取

def extract_features(text, model=None):
    """返回 dict: 特征名 -> 数值。model 用于查连接词表（train 时传 None）。"""
    sents = split_sentences(text)
    sent_lens = [cjk_len(s) for s in sents]
    total = max(cjk_len(text), 1)
    feats = {}

    # 1. 句长分布
    if len(sent_lens) >= MIN_SENTENCES:
        feats["sent_len_std"] = statistics.pstdev(sent_lens)
        feats["sent_len_mean"] = statistics.mean(sent_lens)
        feats["sent_len_cv"] = feats["sent_len_std"] / max(feats["sent_len_mean"], 1)
    else:
        feats["sent_len_std"] = feats["sent_len_mean"] = feats["sent_len_cv"] = None

    # 2. 标点习惯（每百字）
    feats["comma_rate"] = text.count("，") / total * 100
    feats["enum_rate"] = text.count("、") / total * 100
    feats["semicolon_rate"] = text.count("；") / total * 100
    feats["colon_rate"] = text.count("：") / total * 100

    # 3. 用字习惯（每百字）
    for ch, name in [("其", "qi_rate"), ("该", "gai_rate"), ("的", "de_rate"),
                     ("了", "le_rate"), ("性", "xing_rate"), ("等", "deng_rate")]:
        feats[name] = text.count(ch) / total * 100

    # 4. 连接词 / 套话密度（每百字，命中哪些词在解释时列出）
    conn_list = model["connectives"] if model else CONNECTIVE_CANDIDATES
    hits = []
    count = 0
    for w in conn_list:
        c = text.count(w)
        if c:
            hits.append((w, c))
            count += c
    feats["connective_rate"] = count / total * 100
    feats["_connective_hits"] = hits

    # 5. 四字格密度（每百字）
    feats["fourchar_rate"] = len(four_char_chunks(text)) / total * 100

    # 5b. AI 腔高频词密度（每百字）
    if model and model.get("style_words"):
        hits = []
        count = 0
        for w in model["style_words"]:
            c = text.count(w)
            if c:
                hits.append((w, c))
                count += c
        feats["style_word_rate"] = count / total * 100
        feats["_style_word_hits"] = hits
    elif not model:
        feats["style_word_rate"] = None

    # 6. AI 偏向特征串密度：训练时从语料统计出的高 log-odds 4-gram 集合
    if model and model.get("ai_ngrams"):
        ai_set = set(model["ai_ngrams"])
        grams = char_ngrams(text, 4)
        hits = [g for g in grams if g in ai_set]
        feats["ai_ngram_rate"] = len(hits) / max(len(grams), 1) * 100
        feats["_ai_ngram_hits"] = sorted(set(hits))
    elif not model:
        feats["ai_ngram_rate"] = None

    return feats


# ---------------------------------------------------------------- 训练

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def train_ngram(texts, n, k):
    counts = {}
    total = 0
    for t in texts:
        for g in char_ngrams(t, n):
            counts[g] = counts.get(g, 0) + 1
            total += 1
    return {"n": n, "k": k, "counts": counts, "total": total}


def ngram_loglik(lm, text):
    n, k = lm["n"], lm["k"]
    counts, total = lm["counts"], lm["total"]
    grams = char_ngrams(text, n)
    if not grams:
        return 0.0, 0
    vocab = max(len(counts), 1)
    ll = 0.0
    for g in grams:
        ll += math.log((counts.get(g, 0) + k) / (total + k * vocab))
    return ll, len(grams)


def cmd_train():
    rows = load_jsonl(TRAIN_PATH)
    texts = {"human": [r["text"] for r in rows if r["label"] == "human"],
             "ai": [r["text"] for r in rows if r["label"] == "ai"]}

    # 从语料里自动筛连接词：只保留两类间每千字频率差足够大的候选
    conn_stats = {}
    for w in CONNECTIVE_CANDIDATES:
        rates = {}
        for lab in ("human", "ai"):
            tot = sum(cjk_len(t) for t in texts[lab])
            rates[lab] = sum(t.count(w) for t in texts[lab]) / tot * 1000
        conn_stats[w] = rates
    connectives = [w for w, r in conn_stats.items()
                   if abs(r["ai"] - r["human"]) >= 0.5 and max(r.values()) >= 0.5]

    # 从语料筛 AI 腔高频词：AI 类每百字频率明显高于 human 类
    style_words = []
    for w in STYLE_WORD_CANDIDATES:
        rates = {}
        for lab in ("human", "ai"):
            tot = sum(cjk_len(t) for t in texts[lab])
            rates[lab] = sum(t.count(w) for t in texts[lab]) / tot * 100
        if rates["ai"] - rates["human"] >= 0.05 and rates["ai"] >= 0.1:
            style_words.append(w)

    # 从语料统计 AI 偏向的字符 4-gram（log-odds 最高、AI 类至少出现 3 次）
    from collections import Counter
    cnt = {lab: Counter() for lab in ("human", "ai")}
    tot = {lab: 0 for lab in ("human", "ai")}
    for lab in ("human", "ai"):
        for t in texts[lab]:
            for g in char_ngrams(t, 4):
                cnt[lab][g] += 1
                tot[lab] += 1
    vocab = len(set(cnt["human"]) | set(cnt["ai"]))
    logodds = {}
    for g in set(cnt["human"]) | set(cnt["ai"]):
        pa = (cnt["ai"][g] + 0.5) / (tot["ai"] + 0.5 * vocab)
        ph = (cnt["human"][g] + 0.5) / (tot["human"] + 0.5 * vocab)
        logodds[g] = math.log(pa / ph)
    ai_ngrams = [g for g in sorted(logodds, key=lambda g: -logodds[g])
                 if cnt["ai"][g] >= 3][:60]

    model = {"style_words": style_words,
             "ai_ngrams": ai_ngrams,
             "connectives": connectives,
             "connective_stats": conn_stats,
             "ngram": {lab: train_ngram(texts[lab], NGRAM_N, NGRAM_K)
                       for lab in ("human", "ai")},
             "gauss": {}, "labels": ["human", "ai"]}

    # 连续特征的高斯参数（每类 mean/std），逐样本提取
    feat_names = None
    per_label = {"human": [], "ai": []}
    for lab in ("human", "ai"):
        for t in texts[lab]:
            f = extract_features(t, model)
            per_label[lab].append(f)
    feat_names = [k for k in per_label["human"][0]
                  if not k.startswith("_") and per_label["human"][0][k] is not None]
    for name in feat_names:
        model["gauss"][name] = {}
        all_vals = [f[name] for lab in ("human", "ai") for f in per_label[lab]
                    if f[name] is not None]
        pooled_sd = statistics.pstdev(all_vals) or 1e-6
        for lab in ("human", "ai"):
            vals = [f[name] for f in per_label[lab] if f[name] is not None]
            model["gauss"][name][lab] = {
                "mean": statistics.mean(vals), "std": pooled_sd}

    # n-gram 意外度：每个样本在两类 LM 下的每字符对数似然差，作为第 7 个特征
    for lab in ("human", "ai"):
        vals = []
        for t in texts[lab]:
            ll_h, n1 = ngram_loglik(model["ngram"]["human"], t)
            ll_a, n2 = ngram_loglik(model["ngram"]["ai"], t)
            vals.append((ll_a - ll_h) / max(n1, 1))
        model["gauss"]["ngram_llr"] = model["gauss"].get("ngram_llr", {})
        model["gauss"]["ngram_llr"][lab] = {"mean": statistics.mean(vals),
                                            "_vals": vals}
    all_llr = [v for lab in ("human", "ai")
               for v in model["gauss"]["ngram_llr"][lab].pop("_vals")]
    pooled = statistics.pstdev(all_llr) or 1e-6
    for lab in ("human", "ai"):
        model["gauss"]["ngram_llr"][lab]["std"] = pooled

    # 存盘（n-gram 计数表较大，但训练集小，可以接受）
    serial = dict(model)
    serial["ngram"] = {lab: {"n": lm["n"], "k": lm["k"], "total": lm["total"],
                             "counts": lm["counts"]}
                       for lab, lm in model["ngram"].items()}
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(serial, f, ensure_ascii=False)
    print(f"模型已写入 {MODEL_PATH}")
    print(f"入选连接词（{len(connectives)} 个）: {'、'.join(connectives)}")
    print(f"入选 AI 腔高频词（{len(style_words)} 个）: {'、'.join(style_words)}")


def load_model():
    with open(MODEL_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 打分与解释

def gauss_logpdf(x, mu, sd):
    return -0.5 * math.log(2 * math.pi * sd * sd) - (x - mu) ** 2 / (2 * sd * sd)


def score_text(text, model):
    """返回 (score, contributions)。score>0 判 AI。contributions 逐项列出。"""
    feats = extract_features(text, model)
    contributions = []

    for name, params in model["gauss"].items():
        if name == "ngram_llr":
            continue
        x = feats.get(name)
        if x is None:
            continue
        lh = gauss_logpdf(x, params["human"]["mean"], params["human"]["std"])
        la = gauss_logpdf(x, params["ai"]["mean"], params["ai"]["std"])
        contributions.append({
            "feature": name, "value": x, "score": la - lh,
            "human_mean": params["human"]["mean"], "ai_mean": params["ai"]["mean"],
        })

    # n-gram 意外度：直接用语料 LM 的每字符对数似然比
    ll_h, n1 = ngram_loglik(model["ngram"]["human"], text)
    ll_a, _ = ngram_loglik(model["ngram"]["ai"], text)
    llr = (ll_a - ll_h) / max(n1, 1)
    p = model["gauss"]["ngram_llr"]
    contributions.append({
        "feature": "ngram_llr", "value": llr,
        "score": gauss_logpdf(llr, p["ai"]["mean"], p["ai"]["std"])
                 - gauss_logpdf(llr, p["human"]["mean"], p["human"]["std"]),
        "human_mean": p["human"]["mean"], "ai_mean": p["ai"]["mean"],
    })

    total = sum(c["score"] for c in contributions)
    return total, contributions, feats


FEATURE_DESC = {
    "sent_len_std": "句长标准差（字）",
    "sent_len_mean": "平均句长（字）",
    "sent_len_cv": "句长变异系数",
    "comma_rate": "逗号密度（每百字）",
    "enum_rate": "顿号密度（每百字）",
    "semicolon_rate": "分号密度（每百字）",
    "colon_rate": "冒号密度（每百字）",
    "qi_rate": "「其」字频率（每百字）",
    "gai_rate": "「该」字频率（每百字）",
    "de_rate": "「的」字频率（每百字）",
    "le_rate": "「了」字频率（每百字）",
    "xing_rate": "「性」字频率（每百字）",
    "deng_rate": "「等」字频率（每百字）",
    "connective_rate": "连接词/套话密度（每百字）",
    "fourchar_rate": "四字格密度（每百字）",
    "ngram_llr": "n-gram 意外度（每字符对数似然比）",
    "ai_ngram_rate": "AI 偏向特征串密度（每百个 4-gram）",
    "style_word_rate": "AI 腔高频词密度（每百字）",
}


def explain(text, model):
    score, contribs, feats = score_text(text, model)
    label = "ai" if score > 0 else "human"
    lines = []
    lines.append(f"判定：{'AI 写的' if label == 'ai' else '人写的'}"
                 f"（总分 {score:+.2f}，>0 判 AI）")
    lines.append("")
    lines.append("逐项特征贡献（按影响大小排序，正分=像 AI，负分=像人）：")
    for c in sorted(contribs, key=lambda c: -abs(c["score"])):
        name = c["feature"]
        desc = FEATURE_DESC.get(name, name)
        direction = "像 AI" if c["score"] > 0 else "像人"
        lines.append(
            f"  [{c['score']:+6.2f}] {desc}：本文 {c['value']:.2f}"
            f"（语料中人写均值 {c['human_mean']:.2f}，AI 均值 {c['ai_mean']:.2f}）→ {direction}")
        if name == "connective_rate" and feats.get("_connective_hits"):
            hits = "、".join(f"「{w}」×{n}" for w, n in feats["_connective_hits"])
            lines.append(f"          命中连接词：{hits}")
        if name == "style_word_rate" and feats.get("_style_word_hits"):
            hits = "、".join(f"「{w}」×{n}" for w, n in feats["_style_word_hits"])
            lines.append(f"          命中 AI 腔高频词：{hits}")
        if name == "ai_ngram_rate" and feats.get("_ai_ngram_hits"):
            hits = "、".join(f"「{g}」" for g in feats["_ai_ngram_hits"][:10])
            lines.append(f"          命中 AI 腔特征串：{hits}")
    return label, score, "\n".join(lines)


# ---------------------------------------------------------------- 命令

def cmd_eval():
    model = load_model()
    rows = load_jsonl(TEST_PATH)
    correct = 0
    for r in rows:
        score, _, _ = score_text(r["text"], model)
        pred = "ai" if score > 0 else "human"
        ok = pred == r["label"]
        correct += ok
        print(f"{'✓' if ok else '✗'} 真实={r['label']:5s} 预测={pred:5s} 分数={score:+.2f}")
    print(f"\n准确率：{correct}/{len(rows)} = {correct/len(rows):.1%}")


def cmd_predict(text):
    model = load_model()
    _, _, report = explain(text, model)
    print(report)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "train":
        cmd_train()
    elif cmd == "eval":
        cmd_eval()
    elif cmd == "predict":
        if sys.argv[2] == "-f":
            with open(sys.argv[3], encoding="utf-8") as f:
                text = f.read()
        else:
            text = sys.argv[2]
        cmd_predict(text)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
