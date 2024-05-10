# A (basic) deep learning framework in Rust
This is an effort to really learn rust and refresh my knowledge of training neural networks. The API is inspired by Torch.

I've really enjoyed my time using Rust so far. Planning to add many more features and statistical results for common datasets.

Right now, all work is done on the CPU. Once a decent API for backpropagation and gradient descent is settled on, the focus will shift towards acceleration using OpenCL and/or Cuda.


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
