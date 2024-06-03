use crate::optimizers::optimizer::Optimizer;
use ndarray::{ArrayD, ArrayViewD};

pub trait NeuralComponent {
    fn infer(&self, input: ArrayViewD<f32>) -> ArrayD<f32>;

    fn forward(&mut self, input: ArrayViewD<f32>) -> ArrayD<f32>;

    fn backward(&mut self, losses: ArrayViewD<f32>) -> ArrayD<f32>;

    fn apply_gradients(&mut self, optimizer: &dyn Optimizer);

    fn zero_gradients(&mut self);
}
