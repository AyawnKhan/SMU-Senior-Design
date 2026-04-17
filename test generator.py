"""
test generator.py

Converts FinQA dataset examples into promptfoo test cases (promptfoo_tests.yaml).
Each test case includes the table, question, and an assertion that the answer appears
in the model output.

Usage:
  python "test generator.py"               # generates 50 tests from dataset/test.json
  python "test generator.py" --limit 100   # generate more
  python "test generator.py" --split dev   # use dev split instead
"""

import argparse
import json


def table_to_text(table) -> str:
    if isinstance(table, list):
        rows = []
        for row in table:
            if isinstance(row, list):
                rows.append("| " + " | ".join(str(c).strip() for c in row) + " |")
            else:
                rows.append(str(row))
        return "\n".join(rows)
    return str(table)


def generate_promptfoo_tests(input_json: str, output_yaml: str, limit: int) -> None:
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    tests = []
    for ex in data:
        qa       = ex.get("qa", {}) or {}
        question = qa.get("question", "").strip()
        answer   = str(qa.get("answer", "")).strip()
        table    = ex.get("table") or ex.get("table_ori")

        if not question or not table:
            continue

        tests.append({
            "table":    table_to_text(table),
            "question": question,
            "answer":   answer,
        })

        if len(tests) >= limit:
            break

    with open(output_yaml, "w", encoding="utf-8") as f:
        for t in tests:
            f.write("- vars:\n")
            f.write("    table: |\n")
            for line in t["table"].split("\n"):
                f.write(f"      {line}\n")
            # Escape quotes in question/answer for YAML safety
            q_escaped = t["question"].replace('"', '\\"')
            a_escaped = t["answer"].replace('"', '\\"')
            f.write(f'    question: "{q_escaped}"\n')
            f.write(f'    answer: "{a_escaped}"\n')
            # Per-test assertion: model output must contain the expected answer
            if t["answer"]:
                f.write("  assert:\n")
                f.write(f'    - type: contains\n')
                f.write(f'      value: "{a_escaped}"\n')
            f.write("\n")

    print(f"Generated {len(tests)} test cases → {output_yaml}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",  default="test", choices=["train", "dev", "test"])
    ap.add_argument("--limit",  type=int, default=50)
    ap.add_argument("--output", default="promptfoo_tests.yaml")
    args = ap.parse_args()

    input_path = f"dataset/{args.split}.json"
    generate_promptfoo_tests(input_path, args.output, args.limit)
