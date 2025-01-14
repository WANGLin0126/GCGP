# Efficient Graph Condensation via Gaussian Process (GCGP) [Paper](./docs/Efficient_Graph_Condensation_via_Gaussian_Processes.pdf.pdf) [arxiv](https://arxiv.org/abs/2501.02565)



## Abstract
Graph condensation reduces graph sizes while maintaining performance, addressing the scalability challenges of GNNs caused by computational inefficiencies on large datasets. Existing methods often rely on bi-level optimization, requiring extensive GNN training and limiting scalability.

This paper proposes **Graph Condensation via Gaussian Process (GCGP)**, a computationally efficient method that uses a **Gaussian Process (GP)** to estimate predictions from input nodes, eliminating the need for iterative GNN training.

GCGP incorporates a **covariance function** that aggregates local neighborhoods to capture complex node dependencies. Additionally, **Concrete random variables** approximate binary adjacency matrices in a continuous, differentiable form, enabling gradient-based optimization of discrete graph structures.


## Methodology

<div style="text-align: center;">
    <img src="./docs/GC.png" alt="Graph Condensation" style="width:50%; height:auto; display: inline-block;">
</div>

*Figure 1: Graph condensation aims to condense a large graph $G$ to a smaller but informative one $G^{\mathcal{S}}$, so that it will not impact the downstream task, such as the training of the GNN models.*


---
Existing graph condensation methods use a bi-level optimization strategy, where the condensed graph trains the GNN in the inner loop and is updated in the outer loop via a matching loss. This approach is computationally expensive due to the need for repeated GNN training.

To address this limitation, the proposed **Graph Condensation via Gaussian Process (GCGP)** method introduces a framework that integrates a **Gaussian Process (GP)** to enhance efficiency in graph condensation tasks. In this context, the condensed synthetic graph $G^{\mathcal{S}}$ represents the observations for the GP. By combining these observations with prior knowledge of the model, the GP derives the posterior distribution of the outputs, thereby circumventing the need for computationally intensive iterative training.

<div style="text-align: center;">
        <img src="./docs/GCGP.png" alt="GCGP" style="width:90%; height:auto; display: inline-block;">
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

python=3.8.20 \
ogb=1.3.6 \
pytorch=1.12.1\
pyg=2.5.2\
numpy=1.24.3

> **Note**: It is recommended to install the `ogb` package first to avoid potential misrecognition of CUDA devices.


You can also use the following command to install the environment:

```bash
conda env create -f environment.yml
```




### Small Datasets (`Cora`, `Citeseer`, `Pubmed`, `Photo`, `Computers`)

If you want to run the code, you can follow the steps below:

```bash
cd gcgp
```
 Then you can try the folowing command to run the code:
 ```bash
 python main.py --dataset Cora --cond_ratio 0.5 --ridge 0.5 --k 4 --epochs 200 --learn_A 0
```


If you want to reproduce all the results, you can run the following commands:

```bash
sh run.sh
```

The `run.sh` script will execute the GCGP model on the five datasets using all the tuned parameters, and the results will be saved in the `./gcgp/outputs/` folder. The `results.py` script will then collect the final results, which will be stored in the `./gcgp/results.csv` file.


For the generalization experiment, you can run the following commands:

```bash
sh run_generalization.sh
```

The output results will be saved in the `./gcgp/outputs_generalization/` folder. The `results_generalization.py` script will collect the final results, which will be saved in the `./gcgp/results_generalization.csv` file.


For the time evaluation experiments, you can run the following command:

```bash
sh run_time.sh
```

The output results will be saved in the `./gcgp/outputs_time/` folder. 


### Large Datasets (`Ogbn-arxiv` and `Reddit`)


The codes for the `Ogbn-arxiv` and `Reddit` datasets are in the `gcgp_ogb` and `gcgp_reddit` folders, respectively.


#### Ogbn-arxiv dataset
For the `Ogbn-arxiv` dataset, you can run the following commands to reproduce the results:

```bash
cd gcgp_ogb
sh run.sh
python results.py
```

`run.sh` will execute the GCGP model using all the tuned parameters, and the results will be saved in the `./gcgp_ogb/outputs/` folder. The `results.py` script will then collect the final results, which will be stored in the `./gcgp_ogb/results.csv` file.

for the efficiency experimets, you can run the following command:

```bash
sh run_time.sh
```
the training time outputs will be saved in the `./gcgp_ogb/outputs_time/` folder.


#### Reddit dataset
For the `Reddit` dataset, you can run the following commands to reproduce the results:

```bash
cd gcgp_reddit
sh run.sh
python results.py
```

`run.sh` will execute the GCGP model using all the tuned parameters, and the results will be saved in the `./gcgp_reddit/outputs/` folder. The `results.py` script will then collect the final results, which will be stored in the `./gcgp_reddit/results.csv` file.

For the efficiency experimets, you can run the following command:

```bash
sh run_time.sh
```

the training time outputs will be saved in the `./gcgp_reddit/outputs_time/` folder.



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