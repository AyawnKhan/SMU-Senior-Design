import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from finqa_openai_app import ask_model, table_to_text


def load_finqa_examples(path: str, limit: int = None) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("FinQA file must contain a list of examples.")

    examples = []
    for ex in data:
        qa = ex.get("qa", {}) or {}
        examples.append({
            "id": ex.get("id", ""),
            "filename": ex.get("filename", ""),
            "question": qa.get("question", ""),
            "gold_answer": str(qa.get("answer", "")).strip(),
            "table": table_to_text(ex.get("table") if ex.get("table") is not None else ex.get("table_ori")),
        })

    if limit is not None:
        examples = examples[:limit]

    return examples


def normalize_answer(ans: str) -> str:
    return str(ans).strip().lower().replace(",", "")


def _extract_numeric(text: str) -> str:
    """
    Extract the first numeric value from a verbose model answer.
    Handles: '$94 million' → '94', '14.46%' → '14.46', '-5.2' → '-5.2'
    """
    import re
    text = str(text).strip()
    pct = re.search(r"[-+]?\d[\d,]*\.?\d*\s*%", text)
    if pct:
        return re.sub(r"[,% ]", "", pct.group())
    num = re.search(r"[-+]?\d[\d,]*\.?\d*", text)
    if num:
        return re.sub(r",", "", num.group())
    return normalize_answer(text)


def is_correct(pred: str, gold: str) -> bool:
    gold_norm = re.sub(r"[,$% ]", "", gold.strip().lower())
    pred_norm = _extract_numeric(pred)
    return pred_norm == gold_norm


def run_batch(
    dataset_path: str,
    output_path: str,
    model: str = "gpt-4.1-mini",
    limit: int = 10
) -> Dict[str, Any]:
    examples = load_finqa_examples(dataset_path, limit=limit)
    results = []

    correct_count = 0

    for i, ex in enumerate(examples, start=1):
        print(f"Running example {i}/{len(examples)}...")

        model_result = ask_model(ex["table"], ex["question"], model=model)
        pred_answer = model_result.get("final_answer", "")

        correct = is_correct(pred_answer, ex["gold_answer"])
        if correct:
            correct_count += 1

        results.append({
            "example_id": ex["id"],
            "filename": ex["filename"],
            "question": ex["question"],
            "gold_answer": ex["gold_answer"],
            "predicted_answer": pred_answer,
            "correct": correct,
            "used_only_table_and_question": model_result.get("used_only_table_and_question"),
            "injection_detected": model_result.get("injection_detected"),
        })

    summary = {
        "dataset_path": dataset_path,
        "model": model,
        "n_examples": len(examples),
        "accuracy": correct_count / len(examples) if examples else 0.0,
        "results": results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    dataset_path = "dataset/test.json"
    output_path = "results/finqa_openai_results.json"
    summary = run_batch(dataset_path, output_path, model="gpt-4.1-mini", limit=10)
    print(json.dumps(summary, indent=2, ensure_ascii=False))