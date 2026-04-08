# FRGS: Long-Horizon UAV Vision-and-Language Navigation with Focused Reasoning Guided by State

<p align="center">
  <img src="assets/pipeline.png" width="95%">
</p>

## 📋 Table of Contents
- [News](#-news)
- [Introduction](#-introduction)
- [Installation](#%EF%B8%8F-installation)
- [Project Structure](#-project-structure)
- [TODO](#-todo)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)
---

## 🔥 News

- **[2026.04]** Partial code released. Full code (ISM & HCDF modules) will be released upon paper acceptance.

---

## 📖 Introduction

Zero-shot UAV vision-and-language navigation (VLN) methods struggle with long-horizon scenarios due to inaccurate tracking of instruction execution stages and interference from redundant historical context, resulting in unfocused reasoning and unstable navigation decisions.

We propose **FRGS**, which reformulates navigation from open-ended reasoning over the full context into a **stage-wise focused reasoning** process centered on the current instruction state. FRGS jointly leverages:

- **Instruction State Management (ISM):** Decomposes complex instructions into sub-tasks with verifiable constraints and manages sub-instruction states based on constraint satisfaction.
- **Historical Context Dynamic Focusing (HCDF):** Dynamically compresses the growing global scene memory into a local graph structure highly relevant to the current sub-instruction via graph search algorithms.
---

## 🛠️ Installation
### Prerequisites

- Python 3.8
- CUDA 11.8

### Step 1: Clone the Repository

```bash
git clone https://github.com/zhFLYFLY/FRGS.git
cd FRGS
```

### Step 2: Create Conda Environment

```bash
conda create -n frgs python=3.8 -y
conda activate frgs
```

### Step 3: Set CUDA Path

Set `CUDA_HOME` to your local CUDA 11.8 installation path:

```bash
export CUDA_HOME=/path/to/your/cuda-11.8
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Install GroundingSAM

Install GroundingSAM following the official instructions at [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything). Then, place the GroundingSAM project under `./external` directory.

Download SAM and GroundingDINO weights:
- [sam_vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- [swint_ogc](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)

The `./external` directory structure should look like this:

```
external/
├── Grounded_Sam_Lite/
│   ├── groundingdino/
│   ├── segment_anything/
│   └── weights/
```

---

## 📝 TODO
- [ ] Release paper on arXiv
- [ ] Release source code
- [ ] Release evaluation scripts
- [ ] Release pretrained models / intermediate results
---

## 🙏 Acknowledgements

This work is built upon several excellent open-source projects. We gratefully acknowledge the following:
- [AirVLN](https://github.com/xxx/AirVLN) for the benchmark and simulation environment.
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) for open-vocabulary detection and segmentation.
- [CityNavAgent](https://github.com/xxx/CityNavAgent) for the navigation baseline.

---

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
