# Data Card: MGTAB (Multi-Relational Graph-Based Twitter Account Benchmark) with GraphMind Synthetic Extension

## 1. Dataset Overview
* **Dataset Name:** MGTAB (Multi-Relational Graph-Based Twitter Account Benchmark) with GraphMind Synthetic Extension
* **Primary Task:** Binary Social Bot Detection (Bot vs. Human Classification)
* **Dataset Creator/Maintainer:** MGTAB base dataset by GraphDetec; Synthetic Extension constructed for GraphMind Adversarial Evaluation
* **Language:** English / Multi-lingual Twitter user interactions
* **License:** Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)

---

## 2. Dataset Structure & Statistics

### Core Base Dataset (MGTAB)
* **Total Accounts (Nodes):** 10,199 expert-annotated Twitter accounts
* **Target Labels:** Binary (`0`: Human, `1`: Bot)
* **Node Features:**
  * **Profile Property Features:** 20 numerical and categorical metadata attributes (e.g., account age, follower count, following count, status count)
  * **Textual Features:** Dense semantic embeddings derived from user tweets and profile descriptions
* **Edge Types (7 Heterogeneous Relations):**
  1. Follower
  2. Friend
  3. Mention
  4. Reply
  5. Quote
  6. Shared URL
  7. Shared Hashtag

### Synthetic Extension (GraphMind-Style Botnet)
To stress-test detectors against LLM-driven agents capable of mimicking organic human social topologies (small-world connectivity and community structures), a synthetic botnet follower network was integrated into the baseline environment.
* **Nodes ($\vert{}V\vert{}$):** 1,500 synthetic bot accounts
* **Communities:** 50 disjoint functional communities
* **Total Synthetic Edges:** ~4,500 edges
* **Graph Architecture:**
  * **Layer 1 (Intra-Community):** Similarity-ordered Watts–Strogatz (WS) small-world ring lattices ($k_{\text{ring}} \ge 1$, $p = 0.12$, similarity floor $s_{\text{floor}} = 0.85$, internal average path length $\in [2, 5]$ hops)
  * **Layer 2 (Inter-Community Bridge Mesh):** Erdős–Rényi random mesh across communities ($n_c = 50$) with coverage-first endpoint selection, degree cap ($d_{\max} = 8$), similarity gating ($\tau_s = 0.4$), and global mean path calibration ($\in [4.77, 4.80]$ hops)

---

## 3. Dynamic Snapshot & Feature Processing Protocol

To support multi-dimensional detection combining local structural density and longitudinal behavioral trajectories:

1. **Dynamic Graph Snapshots ($\mathcal{G}$):**
   * Interaction graph divided into $T = 10$ sequential temporal snapshots: $\mathcal{G} = \{G^{(1)}, G^{(2)}, \dots, G^{(T)}\}$
   * Each snapshot $G^{(t)} = (V^{(t)}, E^{(t)})$ reflects active follower links and interaction records established up to step $t$.

2. **Structural Density Metrics (Node-Level):**
   * **Degree ($\text{deg}(v)$):** Local immediate connectivity
   * **Core Number ($\text{core}(v)$):** Membership in $k$-core cohesive subgraphs
   * **Truss Number ($\text{truss}(v)$):** Average $k$-truss triangular cohesion surrounding node $v$

3. **Temporal Position & Indicator Signals:**
   * **Absolute Temporal Position Embedding ($p^{\text{AT}}_t$):** Sequential timestamp order
   * **Local Clustering Coefficient ($\text{LCC}_{i,t}$):** Neighborhood interconnectivity ratio
   * **Bidirectional Links Ratio ($\text{BLR}_{i,t}$):** Reciprocity fraction of links at snapshot $t$

---

## 4. Intended Use & Applications

* **Primary Use Case:** Benchmarking and evaluating graph-based machine learning models, temporal neural networks, and structural embedding algorithms against advanced, human-like social botnets.
* **Target Audience:** Cybersecurity researchers, graph machine learning practitioners, and social platform integrity teams.
* **Out-of-Scope Uses:**
  * Automated account suspension or policing without human-in-the-loop verification.
  * Commercial exploitation or redistribution violating CC BY-NC-ND 4.0 terms.
  * Attempting to re-identify, deanonymize, or contact original Twitter users.

---

## 5. Ethical Considerations & Limitations

* **Privacy & Provenance:** BASE MGTAB data consists of publicly available Twitter account metadata and interaction graphs. No personal identification information (PII) extraction or user tracking was conducted.
* **Synthetic Graph Safety:** The GraphMind synthetic botnet component is exclusively a structural topology simulation containing no real platform accounts, automated text generation engines, or active deployment routines.
* **Misclassification Risks:** The proposed hybrid detector achieves a 93.8% Precision and 94.1% Recall. The residual false positive rate (~6.2%) emphasizes that model outputs should be used as decision-support indicators rather than triggers for autonomous account bans.
* **Computational Complexity:** Computing node truss numbers scales as $\mathcal{O}(\Vert{}E\Vert{}^{1.5})$, requiring optimized graph processing for platform-scale deployments.