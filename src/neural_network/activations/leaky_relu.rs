use crate::neural_network::module::Module;
use ndarray::{Array0, ArrayD, ArrayViewD, Axis, IxDyn};

pub struct LeakyRelu {
    negative_slope: f32,

    gradients: ArrayD<f32>,
}

impl LeakyRelu {
    pub fn new(negative_slope: f32) -> Self {
        let gradients = Array0::<f32>::into_dyn(Default::default());

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

    fn prepare(&mut self, batch_size: usize, input_dim: IxDyn) -> IxDyn {
        self.gradients = self
            .gradients
            .clone()
            .into_shape(input_dim.clone())
            .unwrap();

        input_dim
    }

    fn forward(&mut self, z: ArrayViewD<f32>) -> ArrayD<f32> {
        let a = self.infer(z);

        let error = self.derivative(a.view());
        self.gradients
            .push(Axis(0), error.view())
            .expect("failed to push gradients to leakyRelu layer");

        a
    }

    fn backward(&self, loss: ArrayViewD<f32>) -> ArrayD<f32> {
        &loss * &self.gradients
    }
}

impl LeakyRelu {
    fn derivative(&mut self, a: ArrayViewD<f32>) -> ArrayD<f32> {
        a.mapv(|el| if el > 0. { 1. } else { self.negative_slope })
    }
}
