use crate::neural_network::module::Module;
use ndarray::{Array1, ArrayD, ArrayViewD, Axis, IxDyn};
use crate::optimizers::optimizer::Optimizer;

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
        self.gradients = ArrayD::<f32>::zeros(input_dim.clone()).insert_axis(Axis(0));

        input_dim
    }

    fn forward(&mut self, z: ArrayViewD<f32>) -> ArrayD<f32> {
        let a = self.infer(z);

        let error = self.derivative(a.view());
        self.gradients.push(Axis(0), error.view()).unwrap();
        self.gradients = self.gradients.sum_axis(Axis(0));

        a
    }

    fn backward(&mut self, losses: ArrayViewD<f32>) -> ArrayD<f32> {
        let delta = &losses.remove_axis(Axis(0)) * &self.gradients.index_axis(Axis(0), 0);

        self.gradients = ArrayD::<f32>::zeros(self.gradients.raw_dim()).insert_axis(Axis(0));

        delta
    }

    fn apply_gradients(&mut self, optimizer: &Box<dyn Optimizer>) {
        // not trainable, do nothing
    }
}

impl LeakyRelu {
    fn derivative(&mut self, a: ArrayViewD<f32>) -> ArrayD<f32> {
        a.mapv(|el| if el > 0. { 1. } else { self.negative_slope })
    }
}
