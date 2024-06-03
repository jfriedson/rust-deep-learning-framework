use crate::neural_network::neural_component::NeuralComponent;
use crate::optimizers::optimizer::Optimizer;
use ndarray::{Array1, ArrayD, ArrayViewD};

pub struct Sigmoid {
    gradients: ArrayD<f32>,
}

#[allow(unused)]
impl Sigmoid {
    pub fn new() -> Self {
        let gradients = Array1::<f32>::zeros(0).into_dyn();

        Sigmoid { gradients }
    }

    fn derivative(&mut self, a: ArrayViewD<f32>) -> ArrayD<f32> {
        &a * (1. - &a)
    }
}

impl NeuralComponent for Sigmoid {
    fn infer(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        input.mapv(|z| 1. / (1. + (-z).exp()))
    }

    fn forward(&mut self, z: ArrayViewD<f32>) -> ArrayD<f32> {
        let a = self.infer(z);

        self.gradients = self.derivative(a.view());

        a
    }

    fn backward(&mut self, losses: ArrayViewD<f32>) -> ArrayD<f32> {
        &losses * &self.gradients
    }

    fn apply_gradients(&mut self, _optimizer: &dyn Optimizer) {
        // not trainable
    }

    fn zero_gradients(&mut self) {
        let gradient_shape = self.gradients.raw_dim();
        self.gradients = ArrayD::<f32>::zeros(gradient_shape);
    }
}
