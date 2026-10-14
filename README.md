# Human-Like Social Bot Detection via Density-Aware Structural Embeddings and Temporal Self-Attention

## Project Overview
This repository contains the complete research codebase, methodology, and experimental framework for a novel, hybrid social bot detection system designed to overcome the severe performance degradation (“the GraphMind deficit”) that standard Graph Neural Networks (GNNs) experience when confronted with advanced, human-like botnets. 

Traditional GNN detectors rely heavily on the topological unawareness and structural deficiencies of legacy botnets. However, modern adversarial frameworks leverage Large Language Models (LLMs) to simulate highly realistic social networks. In this project, we simulated an advanced botnet follower network and proposed a robust detection architecture that bridges granular local structural density with longitudinal temporal dynamics.

---

## Key Components & Architecture

### 1. Advanced Botnet Simulation & LLM Fine-Tuning
To model realistic human-like evasion strategies, we constructed a synthetic botnet follower network incorporating dense intra-community clusters and a sparse inter-community bridge mesh. 
* **LLM Integration:** We fine-tuned the **Qwen2.5-1.5B-Instruct** base model (`BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"`) using both **Fine-Grained Interaction Modeling (FIM)** and **Graph-Augmented Social Inference (GSI)** paradigms. This equipped the simulated agents to execute context-appropriate actions and construct cross-community follow edges satisfying small-world properties (such as short path lengths and high clustering coefficients) and homophily principles.
* **Two-Layer Synthetic Graph Structure:** The generated botnet comprises 1,500 nodes, 50 communities, and approximately 4,500 edges. Intra-community connections are formed using similarity-ordered Watts–Strogatz processes, while inter-community links are established via a calibrated bridge mesh optimized for small-world path targets and complete coverage.

### 2. Density-Aware Structural Embedding Module (Adapted from DECODE)
To penetrate the structural camouflage of human-like botnets, our pipeline bypasses macro-topological assumptions by focusing on granular local density:
* **Density-Based Random Weighted Walks (RWW):** Computes transition probabilities biased by normalized local density functions incorporating **degree**, **core number**, and **truss number**.
* **Skip-gram Structural Encoding:** Converts the generated density-biased walk sequences into dense structural embeddings ($Z_{\text{struct}}$) that capture local neighborhood properties and functional core memberships.

### 3. Temporal Pattern Extraction Module (Adapted from BotTGCL)
Because bot accounts evolve dynamically over time, our framework tracks longitudinal behavioral trajectories:
* **Dynamic Graph Snapshots:** Partitions the interaction timeline into discrete temporal snapshots to monitor structural and engagement shifts.
* **Multi-Head Temporal Attention Encoder:** Integrates historical context using absolute temporal position embeddings and behavioral indicators (such as Local Clustering Coefficient and Bidirectional Links Ratio) to generate robust temporal embeddings ($Z_{\text{temp}}$).

### 4. Feature Fusion & Classification
* **Concatenation Strategy:** The density-aware structural embeddings ($Z_{\text{struct}}$) and temporal self-attention embeddings ($Z_{\text{temp}}$) are concatenated into a unified hybrid representation vector ($Z_{\text{hybrid}}$).
* **Multi-Layer Perceptron (MLP):** The hybrid representation is passed through an MLP classification network equipped with non-linear activations and dropout regularization to accurately classify accounts as human or bot.

---

## Experimental Highlights & Ablation Studies
* **Mitigating the Simulation Deficit:** Evaluations on benchmark datasets (including MGTAB integrated with our GraphMind-styled simulation) demonstrate that while standard GNNs (GCN, GAT, BotRGCN, RGT) suffer a 15% to 25% drop in performance when facing human-like topologies, our hybrid framework maintains superior resilience and high F1-scores.
* **Ablation Studies:** Comprehensive evaluations confirm that removing individual local density metrics (degree, core, truss) or stripping away the multi-head temporal attention encoder leads to measurable performance degradation, proving that the synergy of structure and time is essential for robust detection.