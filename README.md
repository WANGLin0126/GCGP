# Efficient Graph Condensation via Gaussian Process (GCGP)

📄 [**Read the Paper**](./docs/Efficient_Graph_Condensation_via_Gaussian_Processes.pdf)

---

## 📚 Table of Contents

- [Efficient Graph Condensation via Gaussian Process (GCGP)](#efficient-graph-condensation-via-gaussian-process-gcgp)
  - [📚 Table of Contents](#-table-of-contents)
  - [🧠 Abstract](#-abstract)
  - [🔬 Methodology](#-methodology)
    - [🧪 GCGP: A Simpler Alternative](#-gcgp-a-simpler-alternative)
  - [🛠️ Implementation](#️-implementation)
    - [🔧 Requirements](#-requirements)
    - [📂 Small Datasets (`Cora`, `Citeseer`, `Pubmed`, `Photo`, `Computers`)](#-small-datasets-cora-citeseer-pubmed-photo-computers)
    - [🗂️ Large Datasets (`Ogbn-arxiv` and `Reddit`)](#️-large-datasets-ogbn-arxiv-and-reddit)
      - [🔹 Ogbn-arxiv Dataset](#-ogbn-arxiv-dataset)
      - [🔹 Reddit Dataset](#-reddit-dataset)
  - [📖 Cite Our Paper](#-cite-our-paper)
  - [📄 License](#-license)

---

## 🧠 Abstract

Graph condensation reduces graph sizes while maintaining performance, addressing the scalability challenges of GNNs caused by computational inefficiencies on large datasets. Existing methods often rely on **bi-level optimization**, which requires repeated GNN training and limits scalability.

This paper proposes **Graph Condensation via Gaussian Process (GCGP)** — a computationally efficient method that leverages a **Gaussian Process (GP)** to estimate predictions from input nodes without iterative GNN training.

Key innovations:

- A **covariance function** aggregates local neighborhoods to capture complex node dependencies.
- **Concrete random variables** approximate binary adjacency matrices in a differentiable form, enabling gradient-based optimization of discrete graph structures.

---

## 🔬 Methodology

<div align="center">
  <img src="./docs/GC.png" alt="Graph Condensation" style="width:45%; height:auto;">
</div>

**Figure 1**: Graph condensation condenses a large graph $G$ into a smaller but informative graph $G^{\mathcal{S}}$ that preserves performance on downstream tasks like GNN training.

---

Conventional graph condensation methods use a **bi-level optimization** framework:

- **Inner loop**: Train a GNN on the condensed graph.
- **Outer loop**: Update the condensed graph based on performance loss.

This is **computationally expensive** due to repeated GNN training.

### 🧪 GCGP: A Simpler Alternative

GCGP replaces iterative GNN training with a **Gaussian Process**, treating the condensed synthetic graph $G^{\mathcal{S}}$ as GP observations. The GP combines these with model priors to make predictions on the original graph $G$.

<div align="center">
  <img src="./docs/GCGP.png" alt="GCGP Workflow" style="width:100%; height:auto;">
</div>

**Figure 2**: The GCGP workflow includes:
1. Using the condensed graph $G^{\mathcal{S}}$ as GP observations.
2. Predicting node labels in the original graph $G$.
3. Optimizing the condensed graph by minimizing the discrepancy between predictions and ground-truth labels.

---

## 🛠️ Implementation

### 🔧 Requirements

- `python=3.8.20`  
- `ogb=1.3.6`  
- `pytorch=1.12.1`  
- `pyg=2.5.2`  
- `numpy=1.24.3`  

> 💡 **Tip**: Install `ogb` first to avoid CUDA device recognition issues.

To set up the environment, run:

```bash
conda env create -f environment.yml
```

---

### 📂 Small Datasets (`Cora`, `Citeseer`, `Pubmed`, `Photo`, `Computers`)

Navigate to the `gcgp` folder:

```bash
cd gcgp
```

Run GCGP on a dataset (e.g., `Cora`):

```bash
python main.py --dataset Cora --cond_ratio 0.5 --ridge 0.5 --k 4 --epochs 200 --learn_A 0
```

To reproduce all results:

```bash
sh run.sh
```

- Outputs will be saved in `./gcgp/outputs/`
- Final results collected in `./gcgp/results.csv` via `results.py`

For **generalization experiments**:

```bash
sh run_generalization.sh
```

- Outputs: `./gcgp/outputs_generalization/`
- Results: `./gcgp/results_generalization.csv`

For **efficiency/time evaluation**:

```bash
sh run_time.sh
```

- Outputs: `./gcgp/outputs_time/`

---

### 🗂️ Large Datasets (`Ogbn-arxiv` and `Reddit`)

#### 🔹 Ogbn-arxiv Dataset

Go to the folder:

```bash
cd gcgp_ogb
```

Run GCGP:

```bash
python main.py --dataset ogbn-arxiv --cond_size 90 --ridge 5 --k 2 --epochs 200 --learn_A 0
```

To reproduce all results:

```bash
sh run.sh
```

- Outputs: `./gcgp_ogb/outputs/`
- Results: `./gcgp_ogb/results.csv`

For time analysis:

```bash
sh run_time.sh
```

- Outputs: `./gcgp_ogb/outputs_time/`

---

#### 🔹 Reddit Dataset

Navigate to:

```bash
cd gcgp_reddit
```

Run GCGP:

```bash
python main.py --dataset Reddit --cond_size 77 --ridge 0.1 --k 2 --epochs 270 --learn_A 0
```

To reproduce all results:

```bash
sh run.sh
```

- Outputs: `./gcgp_reddit/outputs/`
- Results: `./gcgp_reddit/results.csv`

For training time evaluation:

```bash
sh run_time.sh
```

- Outputs: `./gcgp_reddit/outputs_time/`

---

## 📖 Cite Our Paper

If you find our paper or code useful, please cite:

```bibtex
@article{wang2025efficient,
  title={Efficient Graph Condensation via Gaussian Process},
  author={Wang, Lin and Li, Qing},
  journal={arXiv preprint arXiv:2501.02565},
  year={2025}
}
```

---

## 📄 License

[MIT License](./LICENSE) © 2025 WANG Lin

