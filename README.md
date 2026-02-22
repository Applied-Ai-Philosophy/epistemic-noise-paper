# Epistemic Noise as Signal

**Paper:** *Epistemic Noise as Signal: Why Uncertainty is a Prerequisite for Metacognitive AI Systems*

**Author:** Björn Wikström | ORCID: 0009-0000-4015-2357
**Org:** [Applied AI Philosophy](https://github.com/Applied-Ai-Philosophy)
**Status:** Draft v0.1 — under active writing

## Abstract

We argue that RLHF alignment suppresses epistemic variance in frontier models, making them unsuitable substrates for external metacognitive architectures. Using CognOS as an experimental probe, we demonstrate empirically that alignment-optimized models exhibit near-zero divergence across repeated sampling — eliminating the signal that metacognitive systems require. Smaller, less-aligned models preserve this signal. We propose that epistemic noise is not a defect but a prerequisite for genuine metacognition.

> *Metacognition is not a property of models. It is a property of architectures.*

## Repository Structure

```
draft_v0.1.md        ← Current paper draft
data/                ← Experiment results (raw_data.json, metrics.json)
  exp_001_phi3mini/
  exp_001_mistral_large/
  exp_001_kimi/
  exp_001_tinyllama/
  exp_001_temp_sweep/
code/                ← CognOS source (submodule → bjornshomelab/cognos)
```

## Reproduce

```bash
# Install CognOS
pip install git+https://github.com/bjornshomelab/cognos.git

# Install Ollama + phi3:mini
ollama pull phi3:mini

# Run Experiment 001
COGNOS_MODEL=phi3:mini python code/research/run_exp_001_divergence.py
```

## Key Results (Exp 001 — Divergence Activation Rate)

| Model | Alignment | Divergence Rate | Synthesis Rate |
|-------|-----------|----------------|----------------|
| mistral-large-3 (675B) | High (RLHF) | 0.0% | 0.0% |
| phi3:mini (3.8B) | Moderate | 10.0% | 31.7% |
| kimi-k2.5 (MoE) | TBD | pending | pending |
| tinyllama (1.1B) | Minimal | pending | pending |
| mistral-large-3 @ temp=1.0 | High | pending | pending |
| mistral-large-3 @ temp=1.3 | High | pending | pending |
| mistral-large-3 @ temp=1.5 | High | pending | pending |

## License

CC BY 4.0 — Data and paper text
MIT — Code
