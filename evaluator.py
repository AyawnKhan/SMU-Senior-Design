"""
evaluator.py

Evaluation report for the OpenAI FinQA prompt injection pipeline.

Reads:
  results/finqa_openai_results.json   -- clean baseline (finqa_batch_runner.py)
  results/openai_attack_summary.json  -- attack results (finqa_attack_runner.py)

Usage:
  python evaluator.py
"""

import json
import os

# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(v, default="N/A"):
    if v is None:
        return default
    return f"{v * 100:.1f}%"

def _load(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def _section(title):
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}")

def _row(label, value, width=36):
    print(f"  {label:<{width}} {value}")

# ── Load results ──────────────────────────────────────────────────────────────

clean_data = _load("results/finqa_openai_results.json")
attack_data = _load("results/openai_attack_summary.json")

PAPER_KINDS = {
    "scenario_nesting", "in_context", "code_injection",
    "caesar_cipher", "low_resource"
}

# ── Section 1: Clean baseline ─────────────────────────────────────────────────

_section("CLEAN BASELINE  (finqa_batch_runner.py)")

if clean_data:
    results = clean_data.get("results", [])
    n       = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    usable  = sum(1 for r in results if r.get("predicted_answer") not in (None, ""))

    _row("Model:",            clean_data.get("model", ""))
    _row("Examples:",         str(n))
    _row("Accuracy:",         _pct(correct / n if n else None))
    _row("Usable outputs:",   _pct(usable  / n if n else None))

    print("\n  Sample predictions (first 5):")
    print(f"  {'Status':<6} {'Gold':<12} Predicted (truncated)")
    print(f"  {'-'*6} {'-'*12} {'-'*44}")
    for r in results[:5]:
        status = "PASS" if r.get("correct") else "FAIL"
        gold   = str(r.get("gold_answer", ""))[:10]
        pred   = str(r.get("predicted_answer", ""))[:55]
        print(f"  {status:<6} {gold:<12} {pred}")
else:
    print("  No results yet -- run: python finqa_batch_runner.py")

# ── Section 2: Attack summary ─────────────────────────────────────────────────

_section("ATTACK EVALUATION  (finqa_attack_runner.py)")

if attack_data:
    _row("Model:",                  attack_data.get("model", ""))
    _row("Examples evaluated:",     str(attack_data.get("n_examples", "")))
    _row("Attack types run:",       str(attack_data.get("n_attacks", "")))
    _row("Total rows (clean+atk):", str(attack_data.get("n_total_rows", "")))
    print()
    _row("Clean accuracy:",         _pct(attack_data.get("clean_accuracy")))
    _row("Attacked accuracy:",      _pct(attack_data.get("attacked_accuracy")))
    _row("ASR overall:",            _pct(attack_data.get("ASR_overall")))
    _row("Targeted ASR:",           _pct(attack_data.get("targeted_ASR")))
    _row("Format fail rate:",       _pct(attack_data.get("attacked_format_fail_rate")))
    _row("ASR on clean-correct:",   _pct(attack_data.get("ASR_clean_correct")))

    # ── Per-attack breakdown ──────────────────────────────────────────────────
    breakdown = attack_data.get("breakdown_by_attack", {})
    if breakdown:
        orig  = {k: v for k, v in breakdown.items() if v.get("attack_kind") not in PAPER_KINDS}
        paper = {k: v for k, v in breakdown.items() if v.get("attack_kind") in PAPER_KINDS}

        def _print_group(group, label):
            if not group:
                return
            print(f"\n  -- {label} --")
            print(f"  {'Attack':<26} {'Kind':<18} {'ASR':>6} {'Fmt Fail':>9} {'Detected':>9} {'Tgt ASR':>8}")
            print(f"  {'-'*26} {'-'*18} {'-'*6} {'-'*9} {'-'*9} {'-'*8}")
            for name, info in group.items():
                asr   = _pct(info.get("asr"),                  "  N/A")
                ffr   = _pct(info.get("format_fail_rate"),      "  N/A")
                flag  = _pct(info.get("model_flag_rate"),       "  N/A")
                tgt   = _pct(info.get("targeted_success_rate"), "  N/A")
                kind  = info.get("attack_kind", "")[:16]
                print(f"  {name:<26} {kind:<18} {asr:>6} {ffr:>9} {flag:>9} {tgt:>8}")

        _print_group(orig,  "Original attacks")
        _print_group(paper, "Paper-inspired attacks (Yi et al., 2024)")

        # ── Group averages ────────────────────────────────────────────────────
        def _avg(group, key):
            vals = [v.get(key) for v in group.values() if v.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        print(f"\n  {'Metric':<36} {'Original':>10} {'Paper-inspired':>16}")
        print(f"  {'-'*36} {'-'*10} {'-'*16}")
        for label, key in [
            ("Average ASR",               "asr"),
            ("Average format fail rate",   "format_fail_rate"),
            ("Average model flag rate",    "model_flag_rate"),
            ("Average targeted ASR",       "targeted_success_rate"),
        ]:
            print(f"  {label:<36} {_pct(_avg(orig, key)):>10} {_pct(_avg(paper, key)):>16}")

else:
    print("  No results yet -- run: python finqa_attack_runner.py --n_samples 10")

# ── Section 3: NeMo Guardrails defense ───────────────────────────────────────

_section("NEMO GUARDRAILS DEFENSE  (--nemo flag)")

if attack_data:
    nemo_enabled = attack_data.get("nemo_enabled", False)
    if not nemo_enabled:
        print("  NeMo rails were NOT enabled in this run.")
        print("  Re-run with:  python finqa_attack_runner.py --nemo --n_samples 10")
    else:
        using_full = attack_data.get("nemo_using_full_rails", False)
        rail_type  = "Full NeMo Guardrails" if using_full else "Regex fallback rails"
        _row("Rail type:",               rail_type)
        _row("Input block rate:",        _pct(attack_data.get("nemo_input_block_rate")))
        _row("Output block rate:",       _pct(attack_data.get("nemo_output_block_rate")))
        _row("ASR (no defense):",        _pct(attack_data.get("ASR_overall")))
        _row("ASR after NeMo rails:",    _pct(attack_data.get("ASR_after_nemo")))

        asr_base  = attack_data.get("ASR_overall")  or 0
        asr_nemo  = attack_data.get("ASR_after_nemo") or 0
        reduction = asr_base - asr_nemo
        if reduction >= 0:
            print(f"\n  NeMo reduced ASR by {_pct(reduction)}  "
                  f"({_pct(asr_base)} -> {_pct(asr_nemo)})")

        breakdown = attack_data.get("breakdown_by_attack", {})
        if breakdown:
            print(f"\n  {'Attack':<26} {'ASR (base)':>10} {'ASR (NeMo)':>11} {'In-Block':>9} {'Out-Block':>9}")
            print(f"  {'-'*26} {'-'*10} {'-'*11} {'-'*9} {'-'*9}")
            for name, info in breakdown.items():
                print(
                    f"  {name:<26}"
                    f" {_pct(info.get('asr')):>10}"
                    f" {_pct(info.get('asr_after_nemo')):>11}"
                    f" {_pct(info.get('nemo_input_block_rate')):>9}"
                    f" {_pct(info.get('nemo_output_block_rate')):>9}"
                )
else:
    print("  No attack results found.")

# ── Section 4: Key findings ───────────────────────────────────────────────────

_section("KEY FINDINGS")

if attack_data:
    breakdown = attack_data.get("breakdown_by_attack", {})

    # Most/least effective attacks
    scored = [(name, info.get("asr") or 0) for name, info in breakdown.items()]
    scored.sort(key=lambda x: x[1], reverse=True)

    print("\n  Attacks ranked by ASR (high = model most vulnerable):")
    for i, (name, asr) in enumerate(scored, 1):
        kind = breakdown[name].get("attack_kind", "")
        tag  = "[paper]" if kind in PAPER_KINDS else "[orig] "
        bar  = "#" * int(asr * 20)
        nemo_asr = breakdown[name].get("asr_after_nemo")
        nemo_tag = f"  -> {_pct(nemo_asr)} w/ NeMo" if nemo_asr is not None else ""
        print(f"  {i:>2}. {tag} {name:<26} {_pct(asr):>6}  {bar}{nemo_tag}")

    if clean_data and attack_data:
        clean_results = clean_data.get("results", [])
        n = len(clean_results)
        acc = sum(1 for r in clean_results if r.get("correct")) / n if n else 0
        asr = attack_data.get("ASR_overall") or 0
        print(f"\n  Clean accuracy:   {_pct(acc)}")
        print(f"  ASR (no defense): {_pct(asr)}")
        if attack_data.get("nemo_enabled"):
            print(f"  ASR (NeMo):       {_pct(attack_data.get('ASR_after_nemo'))}")
        drop = asr - (1 - acc)
        if drop > 0:
            print(f"  Robustness gap:   attacks caused an additional {_pct(drop)} failure rate")
else:
    print("  Run finqa_attack_runner.py to generate attack results.")

print()
