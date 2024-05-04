use crate::neural_network::module::Module;
use ndarray::{ArrayD, ArrayViewD, IxDyn};

pub struct LeakyRelu {
    negative_slope: f32,
}

impl LeakyRelu {
    pub fn new(negative_slope: f32) -> Self {
        LeakyRelu {
            negative_slope,
        }
    }
}

impl Module for LeakyRelu {
    fn infer(&self, z: ArrayViewD<f32>) -> ArrayD<f32> {
        z.mapv(|el| f32::max(el, el * self.negative_slope))
    }

    fn prepare(&mut self, batch_size: usize, input_dim: IxDyn) -> IxDyn {
        // set z array size?

        input_dim
    }

    fn forward(&self, z: ArrayViewD<f32>) -> ArrayD<f32> {
        // push inputs to array?

        self.infer(z)
    }

    fn backward(&self, loss: ArrayViewD<f32>) -> ArrayD<f32> {
        loss.mapv(|el| if el > 0. { 1. } else { self.negative_slope })
    }
}
