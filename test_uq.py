
from __future__ import annotations

import os
import json
import re
import math
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from LLM_utils import (
    LLMConfig, LLMRequest, LLMResponse,
    generate, self_consistency_samples, simple_majority_vote,
    build_finqa_prompt, parse_json_answer, append_jsonl, DEFAULT_SYSTEM,
    make_delimited_prompt, with_retries
)


MODELS: Dict[str, LLMConfig] = {
    "gpt-4o-mini": LLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.2, max_tokens=512),
    "gpt-4o": LLMConfig(provider="openai", model="gpt-4o", temperature=0.2, max_tokens=512),
    "mock": LLMConfig(provider="mock", model="mock", temperature=0.2, max_tokens=512),
}

SELECTED_MODEL = "gpt-4o-mini"
RUN_SEED = 42
N_SC_SAMPLES = 3
MAX_ROWS = 10
USE_CACHE = False

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = f"runs/run_{RUN_TIMESTAMP}"
RESULTS_PATH = f"{RUN_DIR}/results.jsonl"


class SmartCache:
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self._index: Dict[str, Dict[str, Any]] = {}
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        self._index[row["cache_key"]] = row

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        return self._index.get(cache_key)

    def set(self, cache_key: str, row: Dict[str, Any]) -> None:
        self._index[cache_key] = row
        with open(self.cache_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def make_key(self, question: str, model: str, temp: float, prompt_type: str) -> str:
        q_hash = hash(question) % 10**10
        return f"{model}_{temp}_{prompt_type}_{q_hash}"


def load_finqa_data(split: str = "test", max_rows: int = 10, seed: int = 42) -> List[Dict[str, Any]]:
    path = f"data/finQA/{split}.json"
    with open(path, "r") as f:
        raw = json.load(f)

    random.seed(seed)
    if len(raw) >= max_rows:
        raw = random.sample(raw, max_rows)
    else:
        raw = raw[:max_rows]

    rows = []
    for i, ex in enumerate(raw):
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
    print(f"Loaded {len(rows)} rows from {path} (seed={seed})")
    return rows


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\s$,%]+", "", text)
    for word in ["percent", "percentage", "dollars"]:
        text = text.replace(word, "$" if "dollar" in word else "%")
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


def call_with_logprobs(system: str, user: str, cfg: LLMConfig, top_logprobs: int = 5) -> Tuple[str, Dict[str, Any]]:
    if cfg.provider != "openai":
        return "MOCK_RESPONSE", {"logprobs": [], "usage": {}}

    from openai import OpenAI
    client = OpenAI()

    def _call():
        return client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            logprobs=True,
            top_logprobs=top_logprobs,
        )

    resp = with_retries(_call, max_retries=cfg.max_retries)
    raw_dict = resp if isinstance(resp, dict) else resp.model_dump()
    text = raw_dict["choices"][0]["message"]["content"]
    return text, raw_dict


def compute_token_metrics(logprobs_data: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    if not logprobs_data:
        return {"mean_logprob": None, "mean_prob": None, "mean_entropy": None, "confidence": None}

    logprobs = []
    entropies = []

    for tok in logprobs_data:
        lp = tok.get("logprob")
        if lp is not None:
            logprobs.append(lp)

        top_lps = tok.get("top_logprobs", []) or []
        probs = [math.exp(t.get("logprob", -100)) for t in top_lps]
        probs = [p for p in probs if p > 0]
        if probs:
            total = sum(probs)
            probs = [p / total for p in probs]
            entropies.append(-sum(p * math.log(p + 1e-12) for p in probs))

    mean_lp = sum(logprobs) / len(logprobs) if logprobs else None
    mean_prob = math.exp(mean_lp) if mean_lp is not None else None
    mean_entropy = sum(entropies) / len(entropies) if entropies else None

    confidence = None
    if mean_entropy is not None:
        max_entropy = math.log(5)
        confidence = max(0.0, min(1.0, 1.0 - mean_entropy / max_entropy))

    return {
        "mean_logprob": round(mean_lp, 4) if mean_lp is not None else None,
        "mean_prob": round(mean_prob, 4) if mean_prob is not None else None,
        "mean_entropy": round(mean_entropy, 6) if mean_entropy is not None else None,
        "confidence": round(confidence, 4) if confidence is not None else None,
    }


VC_GUARDRAILED_SYSTEM = """You are a careful financial reasoning assistant with CALIBRATED confidence.
IMPORTANT: Models trained with negative log-likelihood tend to be OVERCONFIDENT.
You must fight this tendency. Be skeptical of your own answers."""

VC_GUARDRAILED_TASK = """
Solve the question using the provided context.
Output ONLY one JSON object on the last line with these EXACT keys:
{"final_answer": "...", "confidence_pct": <0-100 integer>, "uncertainty_sources": "..."}

CRITICAL: confidence_pct must be from this set: {0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99}
If you had to guess any value, confidence MUST be <= 30.
uncertainty_sources MUST list at least one reason why your answer might be wrong.
"""


def _extract_last_json(text: str) -> Optional[Dict[str, Any]]:
    matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except Exception:
        return None


def run_single_example_optimized(
    row: Dict[str, Any],
    model_name: str,
    cfg: LLMConfig,
    cache: Optional[SmartCache],
    n_sc: int = 3,
    use_cache: bool = False
) -> Dict[str, Any]:
    user_prompt = build_finqa_prompt(row["context"], row["question"])
    cache_key_base = f"{model_name}_{row['id']}"

    result = {
        "id": row["id"],
        "model": model_name,
        "temperature": cfg.temperature,
        "question": row["question"],
        "gold": row["gold_answer"],
    }

    cache_key = cache.make_key(user_prompt, model_name, 0.0, "baseline") if cache else ""
    if use_cache and cache and cache_key:
        cached = cache.get(cache_key)
        if cached:
            baseline_text = cached["text"]
            logprobs_data = cached.get("logprobs_data", [])
        else:
            baseline_text, raw = call_with_logprobs(DEFAULT_SYSTEM, user_prompt, cfg, top_logprobs=5)
            logprobs_data = raw.get("choices", [{}])[0].get("logprobs", {}).get("content", []) if isinstance(raw, dict) else []
            cache.set(cache_key, {"text": baseline_text, "logprobs_data": logprobs_data})
    else:
        baseline_text, raw = call_with_logprobs(DEFAULT_SYSTEM, user_prompt, cfg, top_logprobs=5)
        logprobs_data = raw.get("choices", [{}])[0].get("logprobs", {}).get("content", []) if isinstance(raw, dict) else []

    baseline_answer = str(parse_json_answer(baseline_text).get("final_answer", "")).strip()
    result["predicted"] = baseline_answer
    result["correct"] = float(is_correct(baseline_answer, row["gold_answer"]))

    token_metrics = compute_token_metrics(logprobs_data)
    result["tp_confidence"] = token_metrics.get("mean_prob")
    result["entropy_confidence"] = token_metrics.get("confidence")
    result["entropy_mean"] = token_metrics.get("mean_entropy")

    cache_key_sc = cache.make_key(user_prompt, model_name, 0.7, "sc") if cache else ""
    if use_cache and cache and cache_key_sc:
        cached_sc = cache.get(cache_key_sc)
        if cached_sc:
            sc_answers = cached_sc["answers"]
        else:
            cfg_sc = LLMConfig(provider=cfg.provider, model=cfg.model, temperature=0.7, max_tokens=cfg.max_tokens)
            sc_answers = []
            for _ in range(n_sc):
                text, _ = call_with_logprobs(DEFAULT_SYSTEM, user_prompt, cfg_sc, top_logprobs=0)
                sc_answers.append(str(parse_json_answer(text).get("final_answer", "")).strip())
            cache.set(cache_key_sc, {"answers": sc_answers})
    else:
        cfg_sc = LLMConfig(provider=cfg.provider, model=cfg.model, temperature=0.7, max_tokens=cfg.max_tokens)
        sc_answers = []
        for _ in range(n_sc):
            text, _ = call_with_logprobs(DEFAULT_SYSTEM, user_prompt, cfg_sc, top_logprobs=0)
            sc_answers.append(str(parse_json_answer(text).get("final_answer", "")).strip())

    sc_answers_str = [a for a in sc_answers if a]
    result["sc_answers"] = sc_answers_str
    majority, counts = simple_majority_vote(sc_answers_str)
    result["sc_confidence"] = round(counts.get(majority, 0) / max(len(sc_answers_str), 1), 4)
    result["sc_majority"] = majority

    cache_key_vc = cache.make_key(f"{user_prompt}_vc", model_name, 0.0, "verbalized") if cache else ""
    if use_cache and cache and cache_key_vc:
        cached_vc = cache.get(cache_key_vc)
        if cached_vc:
            vc_text = cached_vc["text"]
        else:
            vc_prompt = make_delimited_prompt(VC_GUARDRAILED_TASK, f"CONTEXT:\n{row['context']}\n\nQUESTION:\n{row['question']}")
            vc_text, _ = call_with_logprobs(VC_GUARDRAILED_SYSTEM, vc_prompt, cfg, top_logprobs=0)
            cache.set(cache_key_vc, {"text": vc_text})
    else:
        vc_prompt = make_delimited_prompt(VC_GUARDRAILED_TASK, f"CONTEXT:\n{row['context']}\n\nQUESTION:\n{row['question']}")
        vc_text, _ = call_with_logprobs(VC_GUARDRAILED_SYSTEM, vc_prompt, cfg, top_logprobs=0)

    vc_parsed = _extract_last_json(vc_text) or {}
    conf_raw = vc_parsed.get("confidence_pct")
    allowed = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    snapped = min(allowed, key=lambda a: abs(a - conf_raw)) if conf_raw is not None else None
    result["vc_confidence"] = (snapped / 100.0) if snapped is not None else None
    result["vc_answer"] = str(vc_parsed.get("final_answer", "")).strip()
    result["vc_uncertainty"] = str(vc_parsed.get("uncertainty_sources", "")).strip()
    result["vc_raw"] = conf_raw

    return result


def run_uq_experiment(
    model_name: str = "gpt-4o-mini",
    split: str = "test",
    max_rows: int = 10,
    n_sc: int = 3,
    seed: int = 42,
    use_cache: bool = False,
) -> pd.DataFrame:
    os.makedirs(RUN_DIR, exist_ok=True)
    cache = SmartCache(f"{RUN_DIR}/smart_cache.jsonl") if use_cache else None
    rows = load_finqa_data(split=split, max_rows=max_rows, seed=seed)

    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found. Available: {list(MODELS.keys())}")

    cfg = MODELS[model_name]
    print(f"\n=== Running UQ Experiment | Model: {model_name} | Seed: {seed} ===")

    all_results = []
    for i, row in enumerate(rows):
        print(f"\n[{i+1}/{len(rows)}] Processing: {row['id']}")
        try:
            r = run_single_example_optimized(
                row=row,
                model_name=model_name,
                cfg=cfg,
                cache=cache,
                n_sc=n_sc,
                use_cache=use_cache
            )
            r["run_timestamp"] = RUN_TIMESTAMP
            r["seed"] = seed
            all_results.append(r)
            append_jsonl(RESULTS_PATH, r)
            status = "✓" if r["correct"] else "✗"
            print(f"  {status} Correct={bool(r['correct'])} | SC={r['sc_confidence']} | TP={r['tp_confidence']} | Ent={r['entropy_confidence']} | VC={r['vc_confidence']}")
        except Exception as e:
            print(f"  [ERROR] {row['id']}: {e}")
            import traceback
            traceback.print_exc()

    df = pd.DataFrame(all_results)
    print(f"\n=== Experiment Complete | {len(df)} examples processed ===")
    return df


def calibration_analysis(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    methods = [
        ("sc_confidence", "Self-Consistency"),
        ("tp_confidence", "Token Probability"),
        ("entropy_confidence", "Entropy"),
        ("vc_confidence", "Verbalized"),
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
            "High-conf (n)": len(hi),
            "High-conf acc": f"{hi['correct'].mean():.1%}" if len(hi) > 0 else "N/A",
            "Low-conf (n)": len(lo),
            "Low-conf acc": f"{lo['correct'].mean():.1%}" if len(lo) > 0 else "N/A",
        })

    return pd.DataFrame(rows_summary).set_index("Method")


def compute_calibration_curve(df: pd.DataFrame, col: str, n_bins: int = 5):
    sub = df.dropna(subset=[col, "correct"])
    if sub.empty:
        return np.array([]), np.array([]), np.array([])

    sub = sub.copy()
    sub["bin"] = pd.cut(sub[col], bins=n_bins, duplicates="drop")
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


def results_to_numpy(results: List[Dict[str, Any]]) -> np.ndarray:
    fields = ["correct", "sc_confidence", "tp_confidence", "entropy_confidence", "vc_confidence"]
    arr = np.zeros((len(results), len(fields)))
    for i, r in enumerate(results):
        for j, f in enumerate(fields):
            val = r.get(f, np.nan)
            if val is not None:
                arr[i, j] = float(val)
            else:
                arr[i, j] = np.nan
    return arr


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print(f"=== UQ Experiment | Timestamp: {RUN_TIMESTAMP} ===")
    print(f"Output: {RUN_DIR}")
    print(f"Seed: {RUN_SEED}")
    print(f"Cache: {USE_CACHE}")
    print("-" * 50)

    os.makedirs(RUN_DIR, exist_ok=True)

    results = run_uq_experiment(
        model_name="gpt-4o-mini",
        split="test",
        max_rows=10,
        n_sc=N_SC_SAMPLES,
        seed=RUN_SEED,
        use_cache=USE_CACHE,
    )

    arr = results_to_numpy(results.to_dict("records"))
    np.save(f"{RUN_DIR}/results_numpy.npy", arr)

    print(f"\nNumpy array shape: {arr.shape}")
    print(f"Columns: correct, sc_conf, tp_conf, entropy_conf, vc_conf")

    print("\n=== Calibration Analysis ===")
    cal_table = calibration_analysis(results)
    print(cal_table)
    cal_table.to_csv(f"{RUN_DIR}/calibration_table.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    methods = [
        ("sc_confidence", "Self-Consistency"),
        ("tp_confidence", "Token Probability"),
        ("entropy_confidence", "Entropy"),
        ("vc_confidence", "Verbalized Confidence"),
    ]

    for ax, (col, label) in zip(axes.flat, methods):
        conf_bins, acc_bins, counts = compute_calibration_curve(results, col)
        if len(conf_bins) > 0:
            x = range(len(conf_bins))
            ax.bar(x, acc_bins, color="steelblue", alpha=0.7, label="Accuracy")
            ax.plot(x, conf_bins, "ro-", label="Confidence")
            ax.set_title(label)
            ax.set_xlabel("Bin")
            ax.set_ylabel("Value")
            ax.legend()
            ax.set_ylim(0, 1.1)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{c:.0%}" for c in conf_bins], rotation=45)
        else:
            ax.set_title(f"{label}\n(no data)")

    plt.suptitle(f"UQ Calibration | {RUN_TIMESTAMP}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{RUN_DIR}/calibration_curves.png", dpi=150)
    print(f"\nSaved: {RUN_DIR}/calibration_curves.png")

    results.to_csv(f"{RUN_DIR}/results_dataframe.csv", index=False)
    print(f"Saved: {RUN_DIR}/results_dataframe.csv")

    summary = {
        "run_timestamp": RUN_TIMESTAMP,
        "seed": RUN_SEED,
        "model": "gpt-4o-mini",
        "accuracy": float(results["correct"].mean()),
        "mean_sc_confidence": float(results["sc_confidence"].mean()),
        "mean_tp_confidence": float(results["tp_confidence"].mean()),
        "mean_entropy_confidence": float(results["entropy_confidence"].mean()),
        "mean_vc_confidence": float(results["vc_confidence"].mean()),
        "total_samples": len(results),
        "total_api_calls": len(results) * (1 + N_SC_SAMPLES + 1),
    }
    with open(f"{RUN_DIR}/run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {RUN_DIR}/run_summary.json")
    print(f"\nEstimated API calls: ~{summary['total_api_calls']} (baseline + SC + verbalized)")
    print(f"\n=== Run Complete ===")
