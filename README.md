# <img src="assets/evovla_logo.png" width="5%" style="vertical-align: middle;"> EvoVLA: Self-Evolving Vision-Language-Action Model

[![arXiv](https://img.shields.io/badge/arXiv-2511.16166-b31b1b.svg)](https://arxiv.org/abs/2511.16166)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://aigeeksgroup.github.io/EvoVLA/)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/AIGeeksGroup/EvoVLA)
[![Data](https://img.shields.io/badge/HuggingFace-Data-blue)](https://huggingface.co/datasets/AIGeeksGroup/Discoverse-L)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**EvoVLA** is a self-evolving Vision-Language-Action (VLA) framework designed to address stage hallucination in long-horizon manipulation tasks. By integrating **Stage-Aligned Reward (SAR)**, **Pose-Based Object Exploration (POE)**, and **Long-Horizon Memory**, EvoVLA achieves robust performance and sample efficiency in both simulation and real-world environments.

<div align="center">
  <img src="assets/teaser.png" width="100%">
</div>

---

## 📰 News
- **[2025-11-27]** 🚀 We released the paper, code, and the **Discoverse-L** benchmark.
- **[2025-11-27]** 🎥 Check out our [Demo Video](EvoVLA-video.mp4) for visualization of our method.

---

## 🎥 Demo Video

<div align="center">
  <video src="EvoVLA-video.mp4" width="100%" controls></video>
  <br>
  <a href="EvoVLA-video.mp4">Download High-Res Video</a>
</div>

---

## 📝 Abstract

Long-horizon robotic manipulation remains challenging for Vision-Language-Action (VLA) models despite recent progress in zero-shot generalization and Sim2Real transfer. Current VLA models suffer from **stage hallucination**, where agents exploit coarse evaluation signals to shortcut multi-step tasks, reporting high progress without actual task completion. 

We present **EvoVLA**, a self-supervised VLA framework addressing this through three synergistic components:
1. **Stage-Aligned Reward (SAR)**: Uses triplet contrastive learning with Gemini-generated hard negatives to prevent visual shortcuts.
2. **Pose-Based Object Exploration (POE)**: Grounds curiosity in relative object-gripper pose rather than pixels.
3. **Long-Horizon Memory**: Employs selective context and gated fusion to stabilize intrinsic shaping.

Extensive evaluations on **Discoverse-L** (our proposed long-horizon benchmark) demonstrate that EvoVLA improves average success by **10.2%** over the strongest baseline (OpenVLA-OFT), achieves **1.5×** better sample efficiency, and reduces stage hallucination from 38.5% to 14.8%.

---

## 🤖 Methodology

<div align="center">
  <img src="assets/overview.pdf" width="100%">
  <p><em>Figure 1: EvoVLA Overview (Click to open PDF)</em></p>
</div>

EvoVLA consists of three key modules built upon the OpenVLA-OFT backbone:

### 1. Stage-Aligned Reward (SAR)
Leverages Gemini 2.5 Pro generated stage dictionaries (positive, negative, hard-negative triplets) to compute dense, stage-faithful rewards via image-text contrastive scoring. Includes temporal smoothing to filter noise.

### 2. Pose-Based Object Exploration (POE)
Grounds curiosity in the latent manipulation dynamics of relative gripper-object poses. Uses forward/inverse world models to drive exploration based on geometric structure rather than visual distractors.

### 3. Long-Horizon Memory
A selective context memory module that retrieves relevant history via attention, fuses it with current observations using a learned gate, and writes back utility-critical context.

---

## 📊 Results

### Simulation (Discoverse-L)
EvoVLA significantly outperforms baselines across three multi-stage tasks: **Block Bridge** (74 stages), **Stack** (18 stages), and **Jujube-Cup** (19 stages).

| Model | Bridge | Jujube-Cup | Stack | Avg. |
| :--- | :---: | :---: | :---: | :---: |
| Octo | 24.8 | 33.7 | 29.1 | 29.2 |
| OpenVLA | 32.6 | 42.0 | 37.5 | 37.4 |
| $\pi_0$-FAST | 47.4 | 56.9 | 52.8 | 52.4 |
| OpenVLA-OFT | 54.1 | 63.5 | 59.4 | 59.0 |
| **EvoVLA (Ours)** | **65.3** | **72.6** | **69.7** | **69.2** |

<div align="center">
  <img src="assets/simulation.png" width="90%">
  <p><em>Figure 2: Qualitative comparison in simulation</em></p>
</div>

### Real-World Deployment
Deployed on **AIRBOT-Play** robot, achieving **54.6%** average success across Sim2Real transfer and on-robot training tasks.

<div align="center">
  <img src="assets/real_world_comparison.png" width="90%">
  <p><em>Figure 3: Real-world performance comparison</em></p>
</div>

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/AIGeeksGroup/EvoVLA.git
cd EvoVLA

# Create conda environment
conda create -n evovla python=3.10
conda activate evovla

# Install dependencies
pip install -r requirements.txt
```

## 📂 Data Preparation

Download the **Discoverse-L** benchmark data and assets:
```bash
# Download instructions (Example)
python scripts/download_data.py --task all
```

## 🚀 Training

To train EvoVLA on the Discoverse-L benchmark:

```bash
# Example training command
python train.py --config configs/evovla_discoverse.yaml --task bridge
```

## ⚖️ License

This project is licensed under the MIT License.

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@article{liu2025evovla,
  title={EvoVLA: Self-Evolving Vision-Language-Action Model},
  author={Liu, Zeting and Yang, Zida and Zhang, Zeyu and Tang, Hao},
  journal={arXiv preprint arXiv:2511.16166},
  year={2025}
}
```

---
<div align="center">
  <img src="assets/evovla_logo.png" width="100px">
  <p>AIGeeksGroup @ Peking University</p>
</div>
