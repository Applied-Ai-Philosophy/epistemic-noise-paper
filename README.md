# When Alignment Reduces Uncertainty

![Status](https://img.shields.io/badge/Status-Published-success)
![License](https://img.shields.io/badge/License-CC_BY_4.0-blue)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18731535-blue)](https://doi.org/10.5281/zenodo.18731535)
[![PhilPapers](https://img.shields.io/badge/PhilPapers-WIKWAR-purple)](https://philpapers.org/rec/WIKWAR)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0000--4015--2357-green)](https://orcid.org/0009-0000-4015-2357)

**When Alignment Reduces Uncertainty: Epistemic Variance Collapse and Its Implications for Metacognitive AI**

**Björn Wikström** — Independent AI Research / Applied AI Philosophy — 2026

> *Metacognition is not a property of models. It is a property of architectures.*

---

## 📄 Read the Paper

| Format | Link |
| ------ | ---- |
| 📖 Full text (Zenodo) | [doi.org/10.5281/zenodo.18731535](https://doi.org/10.5281/zenodo.18731535) |
| 🔍 PhilPapers record | [philpapers.org/rec/WIKWAR](https://philpapers.org/rec/WIKWAR) |
| 📝 Draft (Markdown) | [draft_v0.1.md](draft_v0.1.md) |

---

## Abstract

Contemporary large language models are optimized for rapid, consistent, and confident output production. We argue that this optimization — achieved primarily through RLHF and related alignment techniques — fundamentally undermines the epistemic conditions required for genuine metacognition in autonomous AI systems.

Using [CognOS](https://github.com/Applied-Ai-Philosophy/cognos) as an experimental probe, we present empirical evidence that alignment-smoothed frontier models exhibit near-zero epistemic variance across repeated sampling, effectively eliminating the uncertainty signal that metacognitive architectures require to function. Smaller, less-aligned models preserve this signal and enable divergence detection, assumption synthesis, and meta-level reasoning that frontier models cannot support.

We propose that **epistemic noise** — variation, uncertainty, and divergence in model outputs — is not a defect to be engineered away, but a necessary prerequisite for metacognitive AI.

---

## Key Results

| Model | Alignment | Divergence Rate | Synthesis Rate |
| ----- | --------- | --------------- | -------------- |
| mistral-large-3 (675B) | High (RLHF) | 0.0% | 0.0% |
| phi3:mini (3.8B) | Moderate | 10.0% | 31.7% |
| tinyllama (1.1B) | Minimal | high | high |

Three epistemic variance profiles emerge:

- **Suppressed** — frontier aligned models, `Ue ≈ 0`, metacognition non-functional
- **Undirected** — minimal alignment, high noise without calibration
- **Calibrated** — medium-scale models, variance tracks question ambiguity

Only *calibrated* variance is epistemically useful for external metacognitive architectures.

---

## Central Claim

Alignment-induced variance collapse creates a **structural incompatibility** between frontier LLMs and external metacognitive architectures. This incompatibility may not be resolvable by scaling — it may require a different design commitment.

The architectural consequence: an external system like CognOS that depends on `Ue > 0` to trigger divergence synthesis is effectively bypassed when built on frontier models. The metacognitive layer becomes decorative.

---

## Experimental Probe: CognOS

All experiments use [CognOS](https://github.com/Applied-Ai-Philosophy/cognos) — an open-source epistemic integrity layer — as the measurement instrument.

```bash
pip install cognos-ai
```

### Reproduce

```bash
# Install Ollama + phi3:mini
ollama pull phi3:mini

# Run Experiment 001 — Divergence Activation Rate
COGNOS_MODEL=phi3:mini python code/run_exp_001_divergence.py
```

---

## Repository

```text
epistemic-noise-paper/
├── draft_v0.1.md       ← Full paper draft (Markdown)
├── data/               ← Raw experiment results
│   ├── exp_001_phi3mini/
│   ├── exp_001_mistral_large/
│   └── exp_001_tinyllama/
└── code/               ← Experiment scripts
```

---

## Citation

```bibtex
@article{wikstrom2026alignment,
  title={When Alignment Reduces Uncertainty: Epistemic Variance Collapse
         and Its Implications for Metacognitive AI},
  author={Wikström, Björn},
  year={2026},
  doi={10.5281/zenodo.18731535},
  url={https://doi.org/10.5281/zenodo.18731535},
}
```

---

## Related

- **CognOS** (experimental probe) — [Applied-Ai-Philosophy/cognos](https://github.com/Applied-Ai-Philosophy/cognos)
- **Applied AI Philosophy** (research ecosystem) — [Applied-Ai-Philosophy](https://github.com/Applied-Ai-Philosophy)

## License

Paper text and data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
Code: MIT
