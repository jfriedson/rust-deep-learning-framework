use crate::neural_network::module::Module;
use ndarray::{Array1, ArrayD, ArrayViewD, Axis, IxDyn};
use crate::optimizers::optimizer::Optimizer;

pub struct Sigmoid {
    gradients: ArrayD<f32>,
}

impl Sigmoid {
    pub fn new() -> Self {
        let gradients = Array1::<f32>::zeros(0).into_dyn();

        Sigmoid { gradients }
    }
}

impl Module for Sigmoid {
    fn infer(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        input.mapv(|z| 1. / (1. + (-z).exp()))
    }

    fn prepare(&mut self, input_dim: IxDyn) -> IxDyn {
        let gradient_shape = input_dim.clone();
        self.gradients = ArrayD::<f32>::zeros(gradient_shape);

        input_dim
    }

    fn forward(&mut self, z: ArrayViewD<f32>) -> ArrayD<f32> {
        let a = self.infer(z);

        let errors = self.derivative(a.view());

        self.gradients += &errors;

        a
    }

    fn backward(&mut self, losses: ArrayViewD<f32>) -> ArrayD<f32> {
        let deltas = &losses * &self.gradients;

        deltas
    }

    fn apply_gradients(&mut self, _optimizer: &Box<dyn Optimizer>) {
        // not trainable, do nothing
    }

    fn zero_gradients(&mut self) {
        let gradient_shape = self.gradients.raw_dim();
        self.gradients = ArrayD::<f32>::zeros(gradient_shape);
    }
}

impl Sigmoid {
    fn derivative(&mut self, a: ArrayViewD<f32>) -> ArrayD<f32> {
        let error = a.mapv(|el| el * (1. - el));

        error
    }
}
