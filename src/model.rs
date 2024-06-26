use crate::neural_network::activations::leaky_relu::LeakyReLU;
use crate::neural_network::activations::softmax::Softmax;
use crate::neural_network::layers::dense::Dense;
use crate::neural_network::neural_component::NeuralComponent;
use crate::optimizers::optimizer::Optimizer;
use ndarray::{ArrayD, ArrayViewD};

pub struct Model {
    dense1: Dense,
    leaky_relu: LeakyReLU,
    dense2: Dense,
    softmax: Softmax,
}

#[allow(unused)]
impl Model {
    pub fn new() -> Self {
        Model {
            dense1: Dense::new(2, 4),
            leaky_relu: LeakyReLU::new(0.1),
            dense2: Dense::new(4, 4),
            softmax: Softmax::new(),
        }
    }

    pub fn infer(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        let z1 = self.dense1.infer(input);
        let a1 = self.leaky_relu.infer(z1.view());

        let z2 = self.dense2.infer(a1.view());
        self.softmax.infer(z2.view())
    }

    pub fn forward(&mut self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        let z1 = self.dense1.forward(input);
        let a1 = self.leaky_relu.forward(z1.view());

        let z2 = self.dense2.forward(a1.view());
        self.softmax.forward(z2.view())
    }

    pub fn backward(&mut self, loss: ArrayViewD<f32>) {
        let a2 = self.softmax.backward(loss);
        let z2 = self.dense2.backward(a2.view());

        let a1 = self.leaky_relu.backward(z2.view());
        self.dense1.backward(a1.view());
    }

    pub fn apply_gradients(&mut self, optimizer: &dyn Optimizer) {
        self.dense1.apply_gradients(optimizer);
        self.dense2.apply_gradients(optimizer);
    }

    pub fn zero_gradients(&mut self) {
        self.dense1.zero_gradients();
        self.leaky_relu.zero_gradients();
        self.dense2.zero_gradients();
        self.softmax.zero_gradients();
    }
}
