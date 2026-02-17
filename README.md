# SMU x Capital One – Trustworthy Gen AI in Financial Services
### A Framework for LLM Hallucination Detection and Adversarial Defense

> **SMU Senior Design Project** | Mentor: [Ranganath Krishnan](https://www.linkedin.com/in/ranganath-krishnan/), Distinguished ML Engineer, AI Labs, FS Tech, Capital One

---

## Team

| Name | GitHub |
|------|--------|
| Ayan Khan | [@AyawnKhan](https://github.com/AyawnKhan) |
| Anum Khan | — |
| Walter Herrera | — |
| Jocelin Macias | — |

---

## 📋 Abstract

As Large Language Models (LLMs) are integrated into high-stakes sectors like finance, their propensity for **hallucinations** (generating plausible but false data) and vulnerability to **adversarial attacks** (prompt injection) pose significant risks.

This project builds a comprehensive framework to quantify model uncertainty and study malicious prompt manipulations. Using the open-source [FinQA dataset](https://arxiv.org/pdf/2109.00122), we develop a suite of tools to ensure LLM outputs are both factually grounded and secure from prompt-injection exploits.

---

## Objectives

- **Hallucination Detection** — Implement and compare state-of-the-art Uncertainty Quantification (UQ) methods to detect hallucinations in financial reasoning.
- **Adversarial Simulation** — Implement and test prompt injection attacks specific to FinQA.
- **Defensive Engineering** — Design and evaluate defense mechanisms (e.g., input sanitization, instruction delimiters, and "Dual-LLM" verification) to mitigate these risks.

---

## Methodology

### Part A: Hallucination Detection via Uncertainty Quantification (UQ)
- Owners: Ayan Khan & Anum Khan
- We explore the hypothesis that a model's *confidence* correlates with its *correctness*. We implement and test existing UQ methods on the FinQA dataset and evaluate whether UQ scores can successfully flag incorrect LLM responses.

### Part B: Cybersecurity & Prompt Injection
- Owners: Walter Herrera & Jocelin Macias
- We simulate a **Red Team (attack) / Blue Team (defense)** environment using frameworks such as [promptfoo](https://www.promptfoo.dev/docs/red-team/quickstart/) to develop and benchmark both attack strategies and defensive countermeasures.

---

## 📦 Deliverables

1. **This GitHub Repository** — Modular code for hallucination detection, prompt injection simulation, and defense mechanisms.
2. **Project Report** — A formal paper detailing the correlation between UQ scores and hallucination rates, and the success rate of various prompt injection and defense mechanisms.

---

## 🗂️ Repository Structure

```
SMU-Senior-Design/
├── hallucination/          # UQ methods and hallucination detection
├── adversarial/            # Prompt injection attack simulations
├── defense/                # Defense mechanism implementations
├── data/                   # FinQA dataset utilities
├── notebooks/              # Experiments and analysis
├── reports/                # Project reports and documentation
└── README.md
```

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/AyawnKhan/SMU-Senior-Design.git
cd SMU-Senior-Design

# Install dependencies
pip install -r requirements.txt
```

---

## 📚 References

| # | Resource |
|---|----------|
| [1] | [FinQA Paper](https://arxiv.org/pdf/2109.00122) · [FinQA GitHub](https://github.com/czyssrs/FinQA) |
| [2] | [LM-Polygraph — UQ for LLMs](https://github.com/IINemo/lm-polygraph) |
| [3] | [Prompt Injection Survey](https://arxiv.org/pdf/2406.15627) · [promptfoo Red Team Quickstart](https://www.promptfoo.dev/docs/red-team/quickstart/) |

---

## 🏫 About

This project is completed as part of the **SMU Lyle School of Engineering Senior Design Program**. The work focuses on two critical dimensions of LLM trustworthiness: **reliability** (hallucination detection) and **cybersecurity** (prompt injection defense).
