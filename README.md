# galerkinNNs
A neural network framework for approximating PDEs with error control. The framework is described in full detail here:

[1] Ainsworth, M., & Dong, J. (2021). [Galerkin neural networks: A framework for approximating variational equations with error control](https://arxiv.org/abs/2105.14094). SIAM Journal on Scientific Computing, 43(4), A2474-A2501.

This implementation is built on [jax](https://github.com/google/jax) and will also require SciPy and matplotlib. Optional packages include joblib (CPU parallelization of linear coefficient updates) and pickle (for saving trained network parameters for later use).
