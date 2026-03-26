
from __future__ import annotations

import os
import json
import re
import math
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from LLM_utils import (
    LLMConfig, LLMRequest, LLMResponse,
    SimpleDiskCache, generate, self_consistency_samples, simple_majority_vote,
    build_finqa_prompt, parse_json_answer, append_jsonl, DEFAULT_SYSTEM,
    make_delimited_prompt, with_retries
)


MODELS: Dict[str, LLMConfig] = {
    "gpt-4o-mini": LLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.2, max_tokens=512),
    "gpt-4o": LLMConfig(provider="openai", model="gpt-4o", temperature=0.2, max_tokens=512),
    "mock": LLMConfig(provider="mock", model="mock", temperature=0.2, max_tokens=512),
}

SELECTED_MODEL = "gpt-4o-mini"
TEMPERATURES = [0.0, 0.2, 0.5, 0.7, 1.0]
N_SC_SAMPLES = 5
MAX_ROWS = 10
RESULTS_PATH = "runs/exploration_results.jsonl"
CACHE_PATH = "runs/llm_cache.jsonl"


def load_finqa_data(split: str = "test", max_rows: int = 10) -> List[Dict[str, Any]]:
    path = f"data/finQA/{split}.json"
    with open(path, "r") as f:
        raw = json.load(f)

    rows = []
    for i, ex in enumerate(raw[:max_rows]):
        pre = " ".join(ex.get("pre_text", []) or [])
        post = " ".join(ex.get("post_text", []) or [])
        table = "\n".join(" | ".join(str(c) for c in r) for r in (ex.get("table") or []))
        rows.append({
            "id": ex.get("id", f"{split}_{i}"),
            "split": split,
            "question": ex["qa"]["question"],
            "context": f"{pre}\n\nTable:\n{table}\n\n{post}".strip(),
            "gold_answer": str(ex["qa"]["answer"]),
        })
    print(f"Loaded {len(rows)} rows from {path}")
    return rows


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\s$,%]+", "", text)
    replacements = {
        "percent": "%",
        "percentage": "%",
        "dollars": "$",
        "dollars": "$",
    }
    for word, sym in replacements.items():
        text = text.replace(word, sym)
    return text


def semantic_match(predicted: str, gold: str) -> bool:
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)
    if pred_norm == gold_norm:
        return True
    try:
        pred_num = float(re.sub(r"[^0-9.-]", "", pred_norm))
        gold_num = float(re.sub(r"[^0-9.-]", "", gold_norm))
        if abs(pred_num - gold_num) / (abs(gold_num) + 1e-9) < 0.01:
            return True
    except (ValueError, ZeroDivisionError):
        pass
    return gold_norm in pred_norm or pred_norm in gold_norm


def is_correct(predicted: str, gold: str) -> bool:
    return semantic_match(predicted, gold)


class NLIClassifier:
    def __init__(self, model_name: str = "microsoft/deberta-v3-large-mnli"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.LABELS = {0: "contradiction", 1: "neutral", 2: "entailment"}

    def classify(self, premise: str, hypothesis: str) -> Tuple[str, Dict[str, float]]:
        import torch

        inputs = self.tokenizer(
            premise, hypothesis,
            truncation=True, max_length=512,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        probs_dict = {self.LABELS[i]: float(probs[i]) for i in range(len(probs))}
        label = self.LABELS[int(np.argmax(probs))]
        return label, probs_dict


nli_classifier = None


def get_nli_classifier() -> NLIClassifier:
    global nli_classifier
    if nli_classifier is None:
        print("Loading NLI classifier (DeBERTa)...")
        nli_classifier = NLIClassifier()
        print("NLI classifier loaded.")
    return nli_classifier


def uq_self_consistency(system: str, user: str, cfg: LLMConfig, n: int = 5, cache: Optional[SimpleDiskCache] = None) -> Dict[str, Any]:
    samples = self_consistency_samples(system, user, cfg, n=n, cache=cache)
    answers = [str(parse_json_answer(s.text).get("final_answer", "")).strip() for s in samples]
    majority, counts = simple_majority_vote(answers)
    confidence = counts.get(majority, 0) / max(len(answers), 1)
    return {
        "method": "self_consistency",
        "confidence": round(confidence, 4),
        "majority_answer": majority,
        "vote_counts": counts,
        "all_answers": answers,
    }


def uq_token_probability(system: str, user: str, cfg: LLMConfig, cache: Optional[SimpleDiskCache] = None) -> Dict[str, Any]:
    if cfg.provider != "openai":
        return {"method": "token_probability", "confidence": None, "mean_logprob": None}

    try:
        from openai import OpenAI
        client = OpenAI()

        def _call():
            return client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=cfg.max_tokens,
                logprobs=True,
                top_logprobs=1,
            )

        resp = with_retries(_call, max_retries=cfg.max_retries)
        lp_data = resp.choices[0].logprobs.content or []
        logprobs = [tok.logprob for tok in lp_data if tok.logprob is not None]

        if not logprobs:
            return {"method": "token_probability", "confidence": None, "mean_logprob": None}

        mean_lp = sum(logprobs) / len(logprobs)
        mean_prob = math.exp(mean_lp)
        confidence = round(mean_prob, 4)

        return {
            "method": "token_probability",
            "mean_logprob": round(mean_lp, 4),
            "confidence": confidence,
            "min_logprob": round(min(logprobs), 4),
            "max_logprob": round(max(logprobs), 4),
        }
    except Exception as e:
        return {"method": "token_probability", "confidence": None, "mean_logprob": None, "error": str(e)}


def uq_entropy(system: str, user: str, cfg: LLMConfig, cache: Optional[SimpleDiskCache] = None) -> Dict[str, Any]:
    if cfg.provider != "openai":
        return {"method": "entropy", "confidence": None, "mean_entropy": None, "normalized_confidence": None}

    try:
        from openai import OpenAI
        client = OpenAI()

        def _call():
            return client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=cfg.max_tokens,
                logprobs=True,
                top_logprobs=5,
            )

        resp = with_retries(_call, max_retries=cfg.max_retries)
        lp_data = resp.choices[0].logprobs.content or []
        entropies = []

        for tok in lp_data:
            probs = [math.exp(t.logprob) for t in (tok.top_logprobs or [])]
            if probs:
                total = sum(probs)
                probs = [p / total for p in probs]
                entropies.append(-sum(p * math.log(p + 1e-12) for p in probs))

        if not entropies:
            return {"method": "entropy", "confidence": None, "mean_entropy": None, "normalized_confidence": None}

        mean_entropy = sum(entropies) / len(entropies)
        max_entropy = math.log(5)
        normalized_conf = 1.0 - (mean_entropy / max_entropy)
        normalized_conf = max(0.0, min(1.0, normalized_conf))

        return {
            "method": "entropy",
            "mean_entropy": round(mean_entropy, 6),
            "confidence": round(normalized_conf, 4),
            "normalized_confidence": round(normalized_conf, 4),
        }
    except Exception as e:
        return {"method": "entropy", "confidence": None, "mean_entropy": None, "normalized_confidence": None, "error": str(e)}


VC_GUARDRAILED_SYSTEM = """You are a careful financial reasoning assistant with CALIBRATED confidence.
IMPORTANT: Models trained with negative log-likelihood tend to be OVERCONFIDENT.
You must fight this tendency. Be skeptical of your own answers.
When uncertain, express lower confidence, NOT higher."""


VC_GUARDRAILED_TASK = """
Solve the question using the provided context.
Output ONLY one JSON object on the last line with these EXACT keys:
{"final_answer": "...", "confidence_pct": <0-100 integer>, "reasoning": "...", "uncertainty_sources": "..."}

CRITICAL CONFIDENCE RULES:
1. confidence_pct must be from this EXACT set: {0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99}
2. If answer requires arithmetic from context values, confidence 50-80
3. If answer relies on text interpretation only, confidence 30-60
4. If multiple reasonable interpretations exist, confidence MUST be <= 40
5. If you had to guess any value, confidence MUST be <= 30
6. NEVER say 90+ unless you explicitly verified every calculation step
7. The uncertainty_sources field MUST list at least one reason why your answer might be wrong
"""


def _extract_last_json(text: str) -> Optional[Dict[str, Any]]:
    matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except Exception:
        return None


def uq_verbalized_guarded(context: str, question: str, cfg: LLMConfig, cache: Optional[SimpleDiskCache] = None) -> Dict[str, Any]:
    user_prompt = make_delimited_prompt(
        VC_GUARDRAILED_TASK,
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
    )
    req = LLMRequest(
        system=VC_GUARDRAILED_SYSTEM,
        user=user_prompt,
        meta={"uq": "verbalized_guarded"},
        config=cfg
    )
    resp = generate(req, cache=cache)
    parsed = _extract_last_json(resp.text) or {}
    conf_raw = parsed.get("confidence_pct", None)

    allowed = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

    def snap(x):
        try:
            x = float(x)
        except Exception:
            return None
        return min(allowed, key=lambda a: abs(a - x))

    snapped = snap(conf_raw)
    return {
        "method": "verbalized_guarded",
        "confidence": (snapped / 100.0) if snapped is not None else None,
        "final_answer": str(parsed.get("final_answer", "")).strip(),
        "vc_raw": conf_raw,
        "vc_snapped": snapped,
        "uncertainty_sources": str(parsed.get("uncertainty_sources", "")).strip(),
        "reasoning": str(parsed.get("reasoning", "")).strip(),
    }


def run_example(
    row: Dict[str, Any],
    model_name: str,
    cfg: LLMConfig,
    n_sc: int = 5,
    cache: Optional[SimpleDiskCache] = None,
    use_nli: bool = True
) -> Dict[str, Any]:
    user_prompt = build_finqa_prompt(row["context"], row["question"])

    baseline = generate(
        LLMRequest(
            system=DEFAULT_SYSTEM,
            user=user_prompt,
            meta={"id": row["id"], "model": model_name},
            config=cfg
        ),
        cache=cache
    )
    baseline_answer = str(parse_json_answer(baseline.text).get("final_answer", "")).strip()
    correct = is_correct(baseline_answer, row["gold_answer"])

    sc = uq_self_consistency(DEFAULT_SYSTEM, user_prompt, cfg, n=n_sc, cache=cache)
    tp = uq_token_probability(DEFAULT_SYSTEM, user_prompt, cfg, cache=cache)
    ent = uq_entropy(DEFAULT_SYSTEM, user_prompt, cfg, cache=cache)
    vc = uq_verbalized_guarded(row["context"], row["question"], cfg, cache=cache)

    nli_label, nli_probs = None, {}
    if use_nli:
        try:
            nli = get_nli_classifier()
            hyp = f"The final answer to the question is: {baseline_answer}"
            nli_label, nli_probs = nli.classify(row["context"], hyp)
        except Exception as e:
            print(f"    [NLI warning] {e}")

    result = {
        "id": row["id"],
        "model": model_name,
        "temperature": cfg.temperature,
        "question": row["question"],
        "gold": row["gold_answer"],
        "predicted": baseline_answer,
        "correct": float(correct),

        "sc_confidence": sc.get("confidence"),
        "sc_majority": sc.get("majority_answer"),
        "sc_vote_ratio": sc.get("confidence"),

        "tp_confidence": tp.get("confidence"),
        "tp_mean_logprob": tp.get("mean_logprob"),

        "entropy_confidence": ent.get("confidence"),
        "entropy_mean": ent.get("mean_entropy"),

        "vc_confidence": vc.get("confidence"),
        "vc_answer": vc.get("final_answer"),
        "vc_uncertainty": vc.get("uncertainty_sources"),

        "nli_label": nli_label,
        "nli_entail_prob": nli_probs.get("entailment"),
        "nli_neutral_prob": nli_probs.get("neutral"),
        "nli_contra_prob": nli_probs.get("contradiction"),
    }
    return result


def results_to_numpy(results: List[Dict[str, Any]]) -> np.ndarray:
    fields = [
        "correct",
        "sc_confidence",
        "tp_confidence",
        "entropy_confidence",
        "vc_confidence",
        "nli_entail_prob",
        "nli_neutral_prob",
        "nli_contra_prob",
    ]
    arr = np.zeros((len(results), len(fields)))
    for i, r in enumerate(results):
        for j, f in enumerate(fields):
            val = r.get(f, np.nan)
            if val is not None:
                arr[i, j] = float(val) if not isinstance(val, (int, float)) else val
            else:
                arr[i, j] = np.nan
    return arr


def run_temperature_experiment(
    model_name: str = "gpt-4o-mini",
    split: str = "test",
    max_rows: int = 10,
    temperatures: List[float] = [0.0, 0.2, 0.5, 0.7, 1.0],
    n_sc: int = 5,
) -> pd.DataFrame:
    cache = SimpleDiskCache(CACHE_PATH)
    rows = load_finqa_data(split=split, max_rows=max_rows)

    all_results = []
    for temp in temperatures:
        print(f"\n=== Temperature = {temp} ===")
        cfg = LLMConfig(
            provider="openai",
            model=MODELS.get(model_name, MODELS["gpt-4o-mini"]).model,
            temperature=temp,
            max_tokens=512
        )

        for row in rows:
            try:
                r = run_example(row, model_name, cfg, n_sc=n_sc, cache=cache, use_nli=False)
                r["temperature"] = temp
                all_results.append(r)
                append_jsonl(RESULTS_PATH, r)
                status = "✓" if r["correct"] else "✗"
                print(f"  {status} [{row['id']}] pred={r['predicted']!r} gold={row['gold_answer']!r}")
            except Exception as e:
                print(f"  [ERROR] {row['id']}: {e}")

    df = pd.DataFrame(all_results)
    print(f"\n=== Temperature Experiment Summary ===")
    summary = df.groupby("temperature")["correct"].agg(["mean", "sum", "count"])
    summary.columns = ["Accuracy", "Correct", "Total"]
    print(summary)
    return df


def run_model_comparison(
    model_names: List[str],
    split: str = "test",
    max_rows: int = 10,
    n_sc: int = 5,
) -> pd.DataFrame:
    cache = SimpleDiskCache(CACHE_PATH)
    rows = load_finqa_data(split=split, max_rows=max_rows)

    all_results = []
    for model_name in model_names:
        if model_name not in MODELS:
            print(f"Warning: {model_name} not in MODELS, skipping")
            continue
        print(f"\n=== Model: {model_name} ===")
        cfg = MODELS[model_name]

        for row in rows:
            try:
                r = run_example(row, model_name, cfg, n_sc=n_sc, cache=cache, use_nli=True)
                all_results.append(r)
                append_jsonl(RESULTS_PATH, r)
                status = "✓" if r["correct"] else "✗"
                print(f"  {status} [{row['id']}] pred={r['predicted']!r} gold={row['gold_answer']!r}")
            except Exception as e:
                print(f"  [ERROR] {row['id']}: {e}")

    df = pd.DataFrame(all_results)

    print("\n=== Model Comparison Summary ===")
    for model_name in model_names:
        sub = df[df["model"] == model_name]
        if sub.empty:
            continue
        acc = sub["correct"].mean()
        mean_conf = sub["sc_confidence"].mean()
        print(f"{model_name}: Accuracy={acc:.1%}, Mean SC Confidence={mean_conf:.2f}")

    return df


def calibration_analysis(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    methods = [
        ("sc_confidence", "Self-Consistency"),
        ("tp_confidence", "Token Probability"),
        ("entropy_confidence", "Entropy"),
        ("vc_confidence", "Verbalized"),
        ("nli_entail_prob", "NLI Entailment"),
    ]

    rows_summary = []
    for col, label in methods:
        sub = df.dropna(subset=[col])
        if sub.empty:
            continue
        hi = sub[sub[col] >= threshold]
        lo = sub[sub[col] < threshold]
        rows_summary.append({
            "Method": label,
            "High-conf n": len(hi),
            "High-conf acc": f"{hi['correct'].mean():.1%}" if len(hi) > 0 else "N/A",
            "Low-conf n": len(lo),
            "Low-conf acc": f"{lo['correct'].mean():.1%}" if len(lo) > 0 else "N/A",
        })

    return pd.DataFrame(rows_summary).set_index("Method")


def compute_calibration_curve(df: pd.DataFrame, col: str, n_bins: int = 10):
    sub = df.dropna(subset=[col, "correct"])
    if sub.empty:
        return np.array([]), np.array([]), np.array([])

    sub = sub.copy()
    sub["bin"] = pd.cut(sub[col], bins=n_bins)
    grouped = sub.groupby("bin", observed=True).agg(
        avg_confidence=("correct", "mean"),
        accuracy=(col, "mean"),
        count=("correct", "count")
    )
    return (
        np.array(grouped["avg_confidence"]),
        np.array(grouped["accuracy"]),
        np.array(grouped["count"])
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    os.makedirs("runs", exist_ok=True)

    results = run_model_comparison(
        model_names=["gpt-4o-mini"],
        split="test",
        max_rows=10,
        n_sc=5,
    )

    arr = results_to_numpy(results.to_dict("records"))
    print(f"\nNumpy results array shape: {arr.shape}")
    print(f"Columns: correct, sc_conf, tp_conf, entropy_conf, vc_conf, nli_entail, nli_neutral, nli_contra")

    print("\n=== Calibration Analysis ===")
    cal_table = calibration_analysis(results)
    print(cal_table)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    methods = [
        ("sc_confidence", "Self-Consistency"),
        ("tp_confidence", "Token Probability"),
        ("entropy_confidence", "Entropy"),
        ("vc_confidence", "Verbalized"),
        ("nli_entail_prob", "NLI Entailment"),
    ]

    for ax, (col, label) in zip(axes.flat, methods):
        conf_bins, acc_bins, counts = compute_calibration_curve(results, col)
        if len(conf_bins) > 0:
            ax.bar(range(len(conf_bins)), acc_bins, color="steelblue", alpha=0.7, label="Actual Accuracy")
            ax.plot(range(len(conf_bins)), conf_bins, "ro-", label="Confidence")
            ax.set_title(label)
            ax.set_xlabel("Bin")
            ax.set_ylabel("Value")
            ax.legend()
            ax.set_ylim(0, 1.1)
        else:
            ax.set_title(f"{label}\n(no data)")

    plt.suptitle("Calibration: Confidence vs Accuracy", fontsize=14)
    plt.tight_layout()
    plt.savefig("runs/calibration_curves.png", dpi=150)
    print("\nSaved calibration curves to runs/calibration_curves.png")
    plt.show()
