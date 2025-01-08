# Efficient Graph Condensation via Gaussian Process (GCGP)


## Abstract
Graph condensation reduces graph sizes while maintaining performance, addressing the scalability challenges of GNNs caused by computational inefficiencies on large datasets. Existing methods often rely on bi-level optimization, requiring extensive GNN training and limiting scalability.

This paper proposes **Graph Condensation via Gaussian Process (GCGP)**, a computationally efficient method that uses a **Gaussian Process (GP)** to estimate predictions from input nodes, eliminating the need for iterative GNN training.

GCGP incorporates a **covariance function** that aggregates local neighborhoods to capture complex node dependencies. Additionally, **Concrete random variables** approximate binary adjacency matrices in a continuous, differentiable form, enabling gradient-based optimization of discrete graph structures.


