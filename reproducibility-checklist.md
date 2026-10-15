# Research Reproducibility & Artifact Checklist

This checklist confirms that all software, computational environments, random seed parameters, and dataset configurations required to reproduce the paper's experimental findings are documented[cite: 2].

---

## 1. Hardware & Operating Environment

* **Operating System:** Ubuntu 22.04 LTS / Linux 5.15 x86_64
* **GPU Compute:** 1× NVIDIA RTX 3090 (24 GB VRAM) / CUDA 11.8
* **CPU Compute:** Intel Core i9-12900K (16 Cores, 24 Threads)
* **System Memory:** 64 GB DDR5 RAM

---

## 2. Software Dependencies & Libraries

```text
python==3.10.12
torch==2.0.1+cu118
torch-geometric==2.3.1
torch-sparse==0.6.17
torch-scatter==2.1.1
networkx==3.1
scikit-learn==1.3.0
numpy==1.24.3
scipy==1.10.1
gensim==4.3.1