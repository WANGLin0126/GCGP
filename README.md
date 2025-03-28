# Efficient Graph Condensation via Gaussian Process (GCGP) [Paper](./docs/Efficient_Graph_Condensation_via_Gaussian_Processes.pdf)



## Abstract
Graph condensation reduces graph sizes while maintaining performance, addressing the scalability challenges of GNNs caused by computational inefficiencies on large datasets. Existing methods often rely on bi-level optimization, requiring extensive GNN training and limiting scalability.

This paper proposes **Graph Condensation via Gaussian Process (GCGP)**, a computationally efficient method that uses a **Gaussian Process (GP)** to estimate predictions from input nodes, eliminating the need for iterative GNN training.

GCGP incorporates a **covariance function** that aggregates local neighborhoods to capture complex node dependencies. Additionally, **Concrete random variables** approximate binary adjacency matrices in a continuous, differentiable form, enabling gradient-based optimization of discrete graph structures.


## Methodology

<div style="text-align: center;">
    <img src="./docs/GC.png" alt="Graph Condensation" style="width:45%; height:auto; display: inline-block;">
</div>

*Figure 1: Graph condensation aims to condense a large graph $G$ to a smaller but informative one $G^{\mathcal{S}}$, so that it will not impact the downstream task, such as the training of the GNN models.*


---
Existing graph condensation methods use a bi-level optimization strategy, where the condensed graph trains the GNN in the inner loop and is updated in the outer loop via a matching loss. This approach is computationally expensive due to the need for repeated GNN training.

To address this limitation, the proposed **Graph Condensation via Gaussian Process (GCGP)** method introduces a framework that integrates a **Gaussian Process (GP)** to enhance efficiency in graph condensation tasks. In this context, the condensed synthetic graph $G^{\mathcal{S}}$ represents the observations for the GP. By combining these observations with prior knowledge of the model, the GP derives the posterior distribution of the outputs, thereby circumventing the need for computationally intensive iterative training.

<div style="text-align: center;">
        <img src="./docs/GCGP.png" alt="GCGP" style="width:100%; height:auto; display: inline-block;">
</div>

*Figure 2: The workflow of the proposed GCGP framework involves three key steps. First, the condensed synthetic graph $G^{\mathcal{S}}$ is utilized as the observations for the GP. Next, predictions are generated for the test locations, corresponding to the original graph $G$. Finally, the condensed graph is iteratively optimized by minimizing the discrepancy between the GP's predictions and the ground-truth labels.*

---

<!-- ## Experimental Results


If you don't want to run the code, We have uploaded the original results of our experiments to the corresponding folders,

- Under `./gcgp/` folder, you can find the original output data for the `Cora`, `Citeseer`, `Pubmed`, `Photo`, and `Computers` datasets. The original outputs are stored in `./gcgp/outputs/`,  `./gcgp/outputs_time/` and `./gcgp/outputs_generalization/`. The node classification accuracy and generalization results are collected in the `./gcgp/results.csv` and `./gcgp/results_generalization.csv`, respectively.  
- Under `./gcgp_ogb/` folder, you can find the original output data for the `ogbn-arxiv` dataset. The original outputs are stored in `./gcgp_ogb/outputs/` and `./gcgp_ogb/outputs_time/`. The node classification accuracy results are collected in the `./gcgp_ogb/results.csv` file.
- Under `./gcgp_reddit/` folder, you can find the original output data for the `Reddit` dataset. The original outputs are stored in `./gcgp_reddit/outputs/` and `./gcgp_reddit/outputs_time/`. The node classification accuracy results are collected in the `./gcgp_reddit/results.csv` file. -->




## Implementation

### Requirements

- `python=3.8.20`  
- `ogb=1.3.6`  
- `pytorch=1.12.1`  
- `pyg=2.5.2`  
- `numpy=1.24.3`  

> **Note**: It is recommended to install the `ogb` package first to avoid potential issues with CUDA device recognition.

You can also use the following command to set up the environment:

```bash
conda env create -f environment.yml
```

---

### Small Datasets (`Cora`, `Citeseer`, `Pubmed`, `Photo`, `Computers`)

First, navigate to the `gcgp` folder, which contains the code for these datasets:

```bash
cd gcgp
```

You can then run the following command to execute the code:

```bash
python main.py --dataset Cora --cond_ratio 0.5 --ridge 0.5 --k 4 --epochs 200 --learn_A 0
```

To reproduce all results, you can simply execute the following script:

```bash
sh run.sh
```

- The `run.sh` script will run the GCGP model on all five datasets using the tuned parameters.  
- The results will be saved in the `./gcgp/outputs/` folder.  
- The `results.py` script will collect final results and store them in the `./gcgp/results.csv` file.

For the generalization experiments, use the following command:

```bash
sh run_generalization.sh
```

- The outputs will be saved in the `./gcgp/outputs_generalization/` folder.  
- The `results_generalization.py` script will collect the final results into the `./gcgp/results_generalization.csv` file.

For the time evaluation experiments, use this command:

```bash
sh run_time.sh
```

- The outputs will be saved in the `./gcgp/outputs_time/` folder.

---

### Large Datasets (`Ogbn-arxiv` and `Reddit`)

The code for the `Ogbn-arxiv` and `Reddit` datasets is located in the `gcgp_ogb` and `gcgp_reddit` folders, respectively.

#### Ogbn-arxiv Dataset

Navigate to the `gcgp_ogb` folder:

```bash
cd gcgp_ogb
```

Run the following command to execute the code:

```bash
python main.py --dataset ogbn-arxiv --cond_size 90 --ridge 5 --k 2 --epochs 200 --learn_A 0
```

To evaluate results across all tuned parameters for the `ogbn-arxiv` dataset, execute the `run.sh` script:

```bash
sh run.sh
```

- The results will be saved in the `./gcgp_ogb/outputs/` folder.  
- Use the `results.py` script to collect the final results, which will be stored in the `./gcgp_ogb/results.csv` file.

For efficiency experiments, run the following command:

```bash
sh run_time.sh
```

- The outputs will be saved in the `./gcgp_ogb/outputs_time/` folder.

#### Reddit Dataset

Navigate to the `gcgp_reddit` folder:

```bash
cd gcgp_reddit
```

Run the following command to execute the code:

```bash
python main.py --dataset Reddit --cond_size 77 --ridge 0.1 --k 2 --epochs 270 --learn_A 0
```

To reproduce all results, execute the `run.sh` script:

```bash
sh run.sh
```

- The `run.sh` script will run the GCGP model using all tuned parameters.  
- The results will be saved in the `./gcgp_reddit/outputs/` folder.  
- The `results.py` script will collect the final results into the `./gcgp_reddit/results.csv` file.

For efficiency experiments, use the following command:

```bash
sh run_time.sh
```

- The training time outputs will be saved in the `./gcgp_reddit/outputs_time/` folder.




## Cite Our Paper
If our paper and codes helps your research, please cite it in your publications:

```bibtex
@article{wang2025efficient,
  title={Efficient Graph Condensation via Gaussian Process},
  author={Wang, Lin and Li, Qing},
  journal={arXiv preprint arXiv:2501.02565},
  year={2025}
}
```