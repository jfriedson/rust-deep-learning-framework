use crate::neural_network::module::Module;
use crate::optimizers::optimizer::Optimizer;
use ndarray::{Array1, ArrayD, ArrayViewD, IxDyn};

pub struct LeakyRelu {
    negative_slope: f32,

    gradients: ArrayD<f32>,
}

impl LeakyRelu {
    pub fn new(negative_slope: f32) -> Self {
        let gradients = Array1::<f32>::zeros(0).into_dyn();

        LeakyRelu {
            negative_slope,

            gradients,
        }
    }
}

impl Module for LeakyRelu {
    fn infer(&self, z: ArrayViewD<f32>) -> ArrayD<f32> {
        z.mapv(|el| f32::max(el, el * self.negative_slope))
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
        let delta = &losses * &self.gradients;

        delta
    }

    fn apply_gradients(&mut self, _optimizer: &Box<dyn Optimizer>) {
        // not trainable, do nothing
    }

    fn zero_gradients(&mut self) {
        let gradient_shape = self.gradients.raw_dim();
        self.gradients = ArrayD::<f32>::zeros(gradient_shape);
    }
}

impl LeakyRelu {
    fn derivative(&mut self, a: ArrayViewD<f32>) -> ArrayD<f32> {
        a.mapv(|el| if el > 0. { 1. } else { self.negative_slope })
    }
}
