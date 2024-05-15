# A (basic) deep learning framework in Rust
This is an effort to really learn rust and refresh my knowledge of training neural networks. The API is inspired by PyTorch.

I've really enjoyed my time using Rust so far. Planning to add many more features and statistical results for common datasets.

Right now, all work is done on the CPU. Now focusing on accelerating computation using OpenCL and/or Cuda.

### Example of training
Training targets:\
![training target](./readme_assets/simple_example-target.png)

Training results (outputs only):\
![training result](./readme_assets/simple_example-result.png)


## Implemented Features
### Layers

- Dense / Fully Connected

### Activations

- Leaky Relu
- Sigmoid

### Loss Functions

- Mean Square Error

### Optimizers

- SGD
- (planned) Adam


## Acceleration Libraries
Currently, only NDArray is used. This package can be accelerated using BLAS.
