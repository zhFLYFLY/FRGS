# FRGS: Long-Horizon UAV Vision-and-Language Navigation with Focused Reasoning Guided by State

<p align="center">
  <img src="assets/pipeline.png" width="95%">
</p>

## 📋 Table of Contents

- [News](#-news)
- [Introduction](#-introduction)
- [TODO](#-todo)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---
## 🔥 News

- **[2026.04]** Repository initialized. Code and models will be released upon paper acceptance.
---

## 📖 Introduction
Zero-shot UAV vision-and-language navigation (VLN) methods struggle with long-horizon scenarios due to inaccurate tracking of instruction execution stages and interference from redundant historical context, resulting in unfocused reasoning and unstable navigation decisions.

We propose **FRGS**, which reformulates navigation from open-ended reasoning over the full context into a **stage-wise focused reasoning** process centered on the current instruction state. FRGS jointly leverages:

- **Instruction State Management (ISM):** Decomposes complex instructions into sub-tasks with verifiable constraints and manages sub-instruction states based on constraint satisfaction.
- **Historical Context Dynamic Focusing (HCDF):** Dynamically compresses the growing global scene memory into a local graph structure highly relevant to the current sub-instruction via graph search algorithms.
---

## 📝 TODO

- [ ] Release paper on arXiv
- [ ] Release source code
- [ ] Release evaluation scripts
- [ ] Release pretrained models / intermediate results

---

## 📝 Citation

If you find this work helpful, please consider citing:

```bibtex
@article{yourname2026frgs,
  title={FRGS: Long-Horizon UAV Vision-and-Language Navigation with Focused Reasoning Guided by State},
  author={Author1 and Author2 and Author3},
  year={2026}
}
