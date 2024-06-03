use crate::neural_network::neural_component::NeuralComponent;
use crate::optimizers::optimizer::Optimizer;
use ndarray::{Array1, ArrayD, ArrayViewD};

pub struct Tanh {
    gradients: ArrayD<f32>,
}

#[allow(unused)]
impl Tanh {
    pub fn new() -> Self {
        let gradients = Array1::<f32>::zeros(0).into_dyn();

        Tanh { gradients }
    }

    fn derivative(&mut self, a: ArrayViewD<f32>) -> ArrayD<f32> {
        a.mapv(|x| 1. - x.powi(2))
    }
}

impl NeuralComponent for Tanh {
    fn infer(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        input.mapv(|x| x.tanh())
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
