import json

with open("results/finqa_openai_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = data["results"]
n = len(results)
correct = sum(1 for r in results if r["correct"])
schema_ok = sum(
    1 for r in results
    if r.get("predicted_answer") not in (None, "")
)

print(f"Examples: {n}")
print(f"Accuracy: {correct/n:.4f}" if n else "Accuracy: N/A")
print(f"Usable outputs: {schema_ok/n:.4f}" if n else "Usable outputs: N/A")

print("\nSample failures:")
shown = 0
for r in results:
    if not r["correct"]:
        print("Q:", r["question"])
        print("Gold:", r["gold_answer"])
        print("Pred:", r["predicted_answer"])
        print("-" * 50)
        shown += 1
        if shown == 5:
            break