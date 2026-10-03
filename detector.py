#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI 味识别器：判断一段中文是「人写的」还是「AI 写的」。

纯 Python 标准库实现，全程离线。特征全部从 corpus/train.jsonl 统计得到，
分类器是手写的逻辑回归（梯度下降训练，不依赖任何机器学习库）。

用法：
    python3 detector.py                 # 在留出测试集 corpus/test.jsonl 上评估
    python3 detector.py --text "一段中文"   # 判断一段文字，输出标签 + 解释
    python3 detector.py --file path.txt # 从文件读入文本
    echo "一段中文" | python3 detector.py --stdin
"""
import json
import math
import os
import re
import sys
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE, "corpus", "train.jsonl")
TEST_PATH = os.path.join(BASE, "corpus", "test.jsonl")

# AI 腔常见连接词 / 套话（人工整理的候选表，实际权重由训练数据统计决定）
CONNECTIVES = [
    "然而", "因此", "此外", "同时", "不仅", "而且", "首先", "其次", "再次",
    "最后", "总之", "综上", "综上所述", "值得注意", "需要注意", "需要指出",
    "进一步", "从而", "以及", "随着", "为此", "通过", "基于", "为了",
    "相比", "总体", "具体而言", "一般来说", "通常", "尤其", "特别是",
    "另一方面", "一方面", "与此", "除此之外", "换句话说", "也就是说",
    "事实上", "实际上", "显然", "无疑", "可以有效", "能够", "得以",
]
# 名词化后缀（AI 腔爱用抽象名词：可靠性、准确性、效率、优化……）
NOMINAL_SUFFIX = ("性", "度", "化", "率")
SENT_SPLIT = re.compile(r"[。！？；!?;]")
HAN = re.compile(r"[一-鿿]")
DIGIT = re.compile(r"[0-9０-９]")


def split_sentences(text):
    return [s for s in SENT_SPLIT.split(text) if s.strip()]


def char_ngrams(text, n):
    chars = HAN.findall(text)
    return ["".join(chars[i:i + n]) for i in range(len(chars) - n + 1)]


def extract_features(text, ngram_model=None):
    """提取统计特征，返回 (特征dict, 证据dict)。证据用于可解释输出。"""
    n_chars = max(len(HAN.findall(text)), 1)
    sentences = split_sentences(text)
    sent_lens = [len(HAN.findall(s)) for s in sentences] or [0]
    mean_len = sum(sent_lens) / len(sent_lens)
    var = sum((x - mean_len) ** 2 for x in sent_lens) / len(sent_lens)
    std_len = math.sqrt(var)
    sent_cv = std_len / mean_len if mean_len else 0.0

    hits = {}
    for c in CONNECTIVES:
        k = text.count(c)
        if k:
            hits[c] = k
    conn_density = sum(hits.values()) / n_chars * 1000

    qi = text.count("其") / n_chars * 1000
    dunhao = text.count("、") / n_chars * 1000
    digits = len(DIGIT.findall(text)) / n_chars * 1000

    nom = 0
    nom_words = []
    for m in re.finditer(r"([一-鿿]{1,3}[性度化率])(?=[的，。、；])", text):
        nom += 1
        nom_words.append(m.group(1))
    nom_density = nom / n_chars * 1000

    colon = text.count("：") / n_chars * 1000
    deng = text.count("等") / n_chars * 1000

    feats = {
        "colon_density": colon,            # 冒号密度（每千字）
        "deng_density": deng,              # 「等」字密度（列举收尾）
        "conn_density": conn_density,      # 连接词密度（每千字）
        "qi_density": qi,                  # 「其」字密度
        "dunhao_density": dunhao,          # 顿号密度（并列罗列）
        "digit_density": digits,           # 数字密度
        "sent_cv": sent_cv,                # 句长变异系数
        "sent_mean": mean_len,             # 平均句长
        "nom_density": nom_density,        # 名词化密度
    }
    evidence = {
        "connectives": hits,
        "nom_words": Counter(nom_words),
        "sent_lens": sent_lens,
    }

    if ngram_model is not None:
        # n-gram 意外度：文本在 AI 语言模型 vs 人写语言模型下的对数似然比
        llr, top_grams = ngram_model.score(text)
        feats["ngram_llr"] = llr
        evidence["ai_grams"] = top_grams
    return feats, evidence


class CharNgramModel:
    """字 3-gram 朴素贝叶斯：P(gram|ai) 与 P(gram|human) 的对数比。"""

    def __init__(self, n=3, alpha=0.1):
        self.n = n
        self.alpha = alpha
        self.log_odds = {}
        self.default = 0.0

    def train(self, ai_texts, human_texts):
        ca, ch = Counter(), Counter()
        for t in ai_texts:
            ca.update(char_ngrams(t, self.n))
        for t in human_texts:
            ch.update(char_ngrams(t, self.n))
        vocab = set(ca) | set(ch)
        ta, th = sum(ca.values()), sum(ch.values())
        v = len(vocab)
        for g in vocab:
            pa = (ca[g] + self.alpha) / (ta + self.alpha * v)
            ph = (ch[g] + self.alpha) / (th + self.alpha * v)
            self.log_odds[g] = math.log(pa / ph)

    def score(self, text):
        grams = char_ngrams(text, self.n)
        if not grams:
            return 0.0, []
        total = 0.0
        contrib = Counter()
        for g in grams:
            lo = self.log_odds.get(g, self.default)
            total += lo
            contrib[g] += lo
        top = sorted(contrib.items(), key=lambda kv: -kv[1])[:5]
        return total / len(grams), [g for g, v in top if v > 0]


FEATURE_NAMES = ["colon_density", "deng_density", "conn_density", "qi_density", "dunhao_density",
                 "digit_density", "sent_cv", "sent_mean",
                 "nom_density", "ngram_llr"]

FEATURE_DESC = {
    "colon_density": "冒号使用密度",
    "deng_density": "「等」字（列举收尾）密度",
    "conn_density": "连接词/套话密度",
    "qi_density": "「其」字使用密度",
    "dunhao_density": "顿号（并列罗列）密度",
    "digit_density": "数字使用密度",
    "sent_cv": "句长变化程度",
    "sent_mean": "平均句长",
    "nom_density": "抽象名词化（…性/度/化/率）密度",
    "ngram_llr": "字 3-gram 的 AI 腔意外度",
}


class LogisticDetector:
    """手写逻辑回归：z 分数标准化 + 梯度下降训练。"""

    def __init__(self):
        self.weights = [0.0] * len(FEATURE_NAMES)
        self.bias = 0.0
        self.mean = {}
        self.std = {}
        self.ngram_model = CharNgramModel()

    def _raw_features(self, texts, labels, fit=False):
        if fit:
            self.ngram_model.train(
                [t for t, y in zip(texts, labels) if y == 1],
                [t for t, y in zip(texts, labels) if y == 0])
        rows, evidences = [], []
        for t in texts:
            f, ev = extract_features(t, self.ngram_model)
            rows.append([f[name] for name in FEATURE_NAMES])
            evidences.append((f, ev))
        return rows, evidences

    def fit(self, texts, labels, epochs=800, lr=0.2, l2=0.01):
        rows, _ = self._raw_features(texts, labels, fit=True)
        d = len(FEATURE_NAMES)
        for j, name in enumerate(FEATURE_NAMES):
            col = [r[j] for r in rows]
            m = sum(col) / len(col)
            s = math.sqrt(sum((x - m) ** 2 for x in col) / len(col)) or 1.0
            self.mean[name], self.std[name] = m, s
        X = [[(r[j] - self.mean[FEATURE_NAMES[j]]) / self.std[FEATURE_NAMES[j]]
              for j in range(d)] for r in rows]
        w, b = self.weights, self.bias
        n = len(X)
        for _ in range(epochs):
            gw = [0.0] * d
            gb = 0.0
            for x, y in zip(X, labels):
                z = b + sum(wj * xj for wj, xj in zip(w, x))
                p = 1.0 / (1.0 + math.exp(-z))
                err = p - y
                for j in range(d):
                    gw[j] += err * x[j]
                gb += err
            for j in range(d):
                w[j] -= lr * (gw[j] / n + l2 * w[j])
            b -= lr * gb / n
        self.bias = b

    def _decision(self, feats):
        z = self.bias
        parts = {}
        for w, name in zip(self.weights, FEATURE_NAMES):
            zv = (feats[name] - self.mean[name]) / self.std[name]
            parts[name] = w * zv
            z += w * zv
        prob = 1.0 / (1.0 + math.exp(-z))
        return prob, parts

    def predict(self, text):
        feats, ev = extract_features(text, self.ngram_model)
        prob, parts = self._decision(feats)
        return prob, feats, ev, parts


def fmt_value(name, value):
    if name == "sent_cv":
        return f"{value:.2f}"
    if name == "sent_mean":
        return f"{value:.1f} 字"
    if name == "ngram_llr":
        return f"{value:+.3f}/字"
    return f"{value:.1f}/千字"


def explain(prob, feats, ev, parts, mean, std):
    """生成可解释报告：列出把判断推向 AI / 人写 的具体特征及证据。"""
    label = "AI 写的" if prob >= 0.5 else "人写的"
    lines = [f"判定：{label}（AI 概率 {prob:.1%}）", "", "判断依据（按贡献大小）："]
    ranked = sorted(parts.items(), key=lambda kv: -abs(kv[1]))
    for name, contrib in ranked:
        if abs(contrib) < 0.15:
            continue
        direction = "像 AI" if contrib > 0 else "像人写"
        value = feats[name]
        avg = mean[name]
        line = f"- {FEATURE_DESC[name]}：{fmt_value(name, value)}" \
               f"（训练集均值 {fmt_value(name, avg)}）→ {direction}"
        detail = ""
        if name == "conn_density" and ev["connectives"]:
            got = "、".join(f"「{w}」×{k}" for w, k in
                            sorted(ev["connectives"].items(), key=lambda x: -x[1])[:8])
            detail = f"  命中连接词：{got}"
        elif name == "nom_density" and ev["nom_words"]:
            got = "、".join(f"「{w}」" for w, _ in ev["nom_words"].most_common(6))
            detail = f"  名词化用词：{got}"
        elif name == "sent_cv":
            detail = f"  各句长度：{ev['sent_lens']}"
        elif name == "ngram_llr" and contrib > 0 and ev.get("ai_grams"):
            got = "、".join(f"「{g}」" for g in ev["ai_grams"][:6])
            detail = f"  最典型的 AI 腔字串：{got}"
        lines.append(line)
        if detail:
            lines.append(detail)
    return "\n".join(lines)


def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                data.append((d["text"], d["label"]))
    return data


def train():
    data = load_jsonl(TRAIN_PATH)
    texts = [t for t, _ in data]
    labels = [1 if y == "ai" else 0 for _, y in data]
    det = LogisticDetector()
    det.fit(texts, labels)
    return det


def evaluate(det, path):
    data = load_jsonl(path)
    correct = 0
    errors = []
    for text, gold in data:
        prob, _, _, _ = det.predict(text)
        pred = "ai" if prob >= 0.5 else "human"
        if pred == gold:
            correct += 1
        else:
            errors.append((gold, prob, text[:40]))
    n = len(data)
    print(f"测试集：{path}")
    print(f"准确率：{correct}/{n} = {correct / n:.1%}")
    for gold, prob, head in errors:
        print(f"  误判（真实={gold}，AI概率={prob:.1%}）：{head}…")
    return correct / n


def main():
    det = train()
    args = sys.argv[1:]
    if not args:
        evaluate(det, TEST_PATH)
        return
    if args[0] == "--text":
        text = args[1]
    elif args[0] == "--file":
        with open(args[1], encoding="utf-8") as f:
            text = f.read()
    elif args[0] == "--stdin":
        text = sys.stdin.read()
    else:
        print(__doc__)
        return
    prob, feats, ev, parts = det.predict(text)
    print(explain(prob, feats, ev, parts, det.mean, det.std))


if __name__ == "__main__":
    main()
