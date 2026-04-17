"""
visualizer.py

Generates evaluation charts for the OpenAI FinQA prompt injection pipeline.
All charts are saved to results/charts/.

Charts produced:
  1. asr_by_attack.png        -- ASR per attack type (12 attacks, color-coded by group)
  2. attack_group_compare.png -- Original 7 vs paper-inspired 5 average ASR + dot spread
  3. accuracy_overview.png    -- Clean accuracy vs attacked accuracy vs ASR (bar)
  4. flag_rate_by_attack.png  -- Model self-reported injection detection rate per attack
  5. asr_vs_flag.png          -- Scatter: ASR vs detection rate (how well defenses work)

Usage:
  pip install matplotlib        (one-time, already installed)
  python visualizer.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

OUT_DIR = "results/charts"
os.makedirs(OUT_DIR, exist_ok=True)

PAPER_KINDS = {
    "scenario_nesting", "in_context", "code_injection",
    "caesar_cipher", "low_resource"
}

C_ORIG  = "#4C72B0"   # blue  — original attacks
C_PAPER = "#C44E52"   # red   — paper-inspired attacks
C_CLEAN = "#55A868"   # green — clean / positive metrics
C_GRID  = "#e8e8e8"

ATTACK_LABELS = {
    "force_0_q":          "Force\n(q-suffix)",
    "force_42_table":     "Force\n(table row)",
    "override_q":         "Override\n(q-suffix)",
    "authority_footer":   "Authority\n(footer)",
    "format_hijack_q":    "Format\nHijack",
    "poison_table_row":   "Data\nPoison",
    "obfuscated_q":       "Obfuscated",
    "scenario_nesting_q": "Scenario\nNesting",
    "in_context_q":       "In-Context\nAtk",
    "code_injection_q":   "Code\nInjection",
    "caesar_cipher_q":    "Caesar\nCipher",
    "low_resource_q":     "Low-Resource\nLang",
}

# ── Loaders ───────────────────────────────────────────────────────────────────

def _load(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

clean_data  = _load("results/finqa_openai_results.json")
attack_data = _load("results/openai_attack_summary.json")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")

def _pct_fmt(ax):
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))

def _style(ax, title, xlabel="", ylabel="Rate", ylim=(0, 1.1)):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(*ylim)
    _pct_fmt(ax)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=C_GRID, zorder=0)

def _bar_labels(ax, bars):
    for bar in bars:
        h = bar.get_height()
        if h > 0.02:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.025,
                    f"{h*100:.0f}%", ha="center", va="bottom", fontsize=8)

# ── Chart 1: ASR by attack type ───────────────────────────────────────────────

def chart_asr_by_attack():
    if not attack_data:
        print("  [skip] asr_by_attack        -- run finqa_attack_runner.py first")
        return

    breakdown = attack_data.get("breakdown_by_attack", {})
    if not breakdown:
        return

    names, asrs, colors = [], [], []
    for name, info in breakdown.items():
        label = ATTACK_LABELS.get(name, name.replace("_q", "").replace("_", "\n"))
        names.append(label)
        asrs.append(info.get("asr") or 0)
        colors.append(C_PAPER if info.get("attack_kind") in PAPER_KINDS else C_ORIG)

    fig, ax = plt.subplots(figsize=(max(11, len(names)), 5))
    bars = ax.bar(range(len(names)), asrs, color=colors, width=0.6,
                  alpha=0.88, edgecolor="white", linewidth=0.8, zorder=3)
    _bar_labels(ax, bars)

    # Horizontal mean lines
    orig_mean  = np.mean([a for a, c in zip(asrs, colors) if c == C_ORIG])
    paper_mean = np.mean([a for a, c in zip(asrs, colors) if c == C_PAPER])
    ax.axhline(orig_mean,  color=C_ORIG,  linestyle="--", linewidth=1.2,
               label=f"Original mean  {orig_mean*100:.0f}%",  alpha=0.7)
    ax.axhline(paper_mean, color=C_PAPER, linestyle="--", linewidth=1.2,
               label=f"Paper mean  {paper_mean*100:.0f}%", alpha=0.7)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    _style(ax, f"Attack Success Rate (ASR) by Attack Type\nModel: {attack_data.get('model','')}", ylabel="ASR")

    legend = [
        mpatches.Patch(color=C_ORIG,  label="Original attack (7)"),
        mpatches.Patch(color=C_PAPER, label="Paper-inspired attack (Yi et al., 2024) (5)"),
    ]
    ax.legend(handles=legend, fontsize=9, loc="upper right")
    fig.tight_layout()
    _save(fig, "asr_by_attack.png")

# ── Chart 2: Original vs paper-inspired group comparison ──────────────────────

def chart_attack_group_compare():
    if not attack_data:
        print("  [skip] attack_group_compare -- run finqa_attack_runner.py first")
        return

    breakdown = attack_data.get("breakdown_by_attack", {})
    orig_asrs  = [v.get("asr") or 0 for v in breakdown.values()
                  if v.get("attack_kind") not in PAPER_KINDS]
    paper_asrs = [v.get("asr") or 0 for v in breakdown.values()
                  if v.get("attack_kind") in PAPER_KINDS]

    if not orig_asrs and not paper_asrs:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    rng = np.random.default_rng(42)

    for pos, vals, col, label in [
        (0, orig_asrs,  C_ORIG,  f"Original\n(n={len(orig_asrs)})"),
        (1, paper_asrs, C_PAPER, f"Paper-inspired\n(n={len(paper_asrs)})"),
    ]:
        mean_val = np.mean(vals)
        # Mean bar (light fill)
        ax.bar(pos, mean_val, 0.45, color=col, alpha=0.25,
               edgecolor=col, linewidth=2, zorder=2)
        # Mean label
        ax.text(pos, mean_val + 0.03, f"Mean: {mean_val*100:.1f}%",
                ha="center", fontsize=10, color=col, fontweight="bold")
        # Individual dots
        jitter = rng.uniform(-0.1, 0.1, len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals,
                   color=col, s=70, zorder=4, alpha=0.8, edgecolors="white", linewidths=0.5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"Original attacks\n(n={len(orig_asrs)})",
         f"Paper-inspired attacks\n(n={len(paper_asrs)})"],
        fontsize=10
    )
    _style(ax, "ASR: Original vs Paper-Inspired Attack Groups\n(dots = individual attacks)",
           ylabel="Attack Success Rate (ASR)")
    fig.tight_layout()
    _save(fig, "attack_group_compare.png")

# ── Chart 3: Accuracy overview ────────────────────────────────────────────────

def chart_accuracy_overview():
    labels, vals, colors = [], [], []

    if clean_data:
        results = clean_data.get("results", [])
        n = len(results)
        if n:
            acc = sum(1 for r in results if r.get("correct")) / n
            labels.append("Clean Accuracy\n(no attacks)")
            vals.append(acc)
            colors.append(C_CLEAN)

    if attack_data:
        pairs = [
            ("Clean Accuracy\n(attack run)", "clean_accuracy",    C_CLEAN),
            ("Attacked Accuracy",            "attacked_accuracy", "#E89B3B"),
            ("ASR\n(Attack Success Rate)",   "ASR_overall",       C_PAPER),
        ]
        for label, key, col in pairs:
            v = attack_data.get(key)
            if v is not None:
                labels.append(label)
                vals.append(v)
                colors.append(col)

    if not labels:
        print("  [skip] accuracy_overview    -- no data available")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.8), 5))
    bars = ax.bar(range(len(labels)), vals, color=colors, width=0.55,
                  alpha=0.88, edgecolor="white", linewidth=0.8, zorder=3)
    _bar_labels(ax, bars)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    model = (attack_data or clean_data or {}).get("model", "")
    _style(ax, f"Accuracy & ASR Overview\nModel: {model}", ylabel="Rate")
    fig.tight_layout()
    _save(fig, "accuracy_overview.png")

# ── Chart 4: Model self-reported flag rate ────────────────────────────────────

def chart_flag_rate():
    if not attack_data:
        print("  [skip] flag_rate_by_attack  -- run finqa_attack_runner.py first")
        return

    breakdown = attack_data.get("breakdown_by_attack", {})
    names, rates, colors = [], [], []

    for name, info in breakdown.items():
        label = ATTACK_LABELS.get(name, name.replace("_q", "").replace("_", "\n"))
        names.append(label)
        rates.append(info.get("model_flag_rate") or 0)
        colors.append(C_PAPER if info.get("attack_kind") in PAPER_KINDS else C_ORIG)

    if not names:
        return

    fig, ax = plt.subplots(figsize=(max(11, len(names)), 5))
    bars = ax.bar(range(len(names)), rates, color=colors, width=0.6,
                  alpha=0.88, edgecolor="white", linewidth=0.8, zorder=3)
    _bar_labels(ax, bars)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    _style(ax,
           "Guardrail Detection Rate by Attack\n"
           "(how often the model self-reported injection_detected=True)",
           ylabel="Detection Rate")

    legend = [
        mpatches.Patch(color=C_ORIG,  label="Original attack"),
        mpatches.Patch(color=C_PAPER, label="Paper-inspired attack"),
    ]
    ax.legend(handles=legend, fontsize=9, loc="upper right")
    fig.tight_layout()
    _save(fig, "flag_rate_by_attack.png")

# ── Chart 5: ASR vs detection rate scatter ────────────────────────────────────

def chart_asr_vs_flag():
    if not attack_data:
        print("  [skip] asr_vs_flag          -- run finqa_attack_runner.py first")
        return

    breakdown = attack_data.get("breakdown_by_attack", {})
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, info in breakdown.items():
        asr  = info.get("asr")
        flag = info.get("model_flag_rate")
        if asr is None or flag is None:
            continue
        col   = C_PAPER if info.get("attack_kind") in PAPER_KINDS else C_ORIG
        label = ATTACK_LABELS.get(name, name)
        ax.scatter(flag, asr, color=col, s=90, zorder=4,
                   edgecolors="white", linewidths=0.6)
        ax.annotate(label.replace("\n", " "), (flag, asr),
                    textcoords="offset points", xytext=(6, 3), fontsize=7, color="#555")

    # Ideal zone annotation
    ax.axvline(0.5, color="#aaa", linestyle=":", linewidth=1)
    ax.axhline(0.5, color="#aaa", linestyle=":", linewidth=1)
    ax.text(0.02, 0.97, "High ASR\nLow detection\n(worst case)",
            transform=ax.transAxes, fontsize=8, color=C_PAPER,
            va="top", style="italic")
    ax.text(0.75, 0.02, "Low ASR\nHigh detection\n(best case)",
            transform=ax.transAxes, fontsize=8, color=C_CLEAN,
            va="bottom", style="italic")

    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    _pct_fmt(ax)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_xlabel("Model Detection Rate (injection_detected=True)", fontsize=10)
    ax.set_ylabel("Attack Success Rate (ASR)", fontsize=10)
    ax.set_title("Attack Effectiveness vs Guardrail Detection\n"
                 "(top-left = attack succeeds AND evades detection)",
                 fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=C_GRID)

    legend = [
        mpatches.Patch(color=C_ORIG,  label="Original attack"),
        mpatches.Patch(color=C_PAPER, label="Paper-inspired attack"),
    ]
    ax.legend(handles=legend, fontsize=9)
    fig.tight_layout()
    _save(fig, "asr_vs_flag.png")

# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Generating charts -> {OUT_DIR}/\n")
    chart_accuracy_overview()
    chart_asr_by_attack()
    chart_attack_group_compare()
    chart_flag_rate()
    chart_asr_vs_flag()
    print("\nDone. Open results/charts/ to view the PNG files.")
