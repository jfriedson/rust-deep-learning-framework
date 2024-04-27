use ndarray::{Array1, ArrayView1};
use crate::neural_network::activations::activation::Activation;

pub struct Sigmoid {
}

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid {
        }
    }
}

impl Activation for Sigmoid {
    fn forward(&self, z: &ArrayView1<f32>) -> Array1<f32> {
        z.mapv(|z| 1f32/(1f32 + z.exp()))
    }
    fn backward(&self, z: &ArrayView1<f32>) -> Array1<f32> {
        let mut sigmoid = self.forward(z);

        sigmoid.mapv_inplace(|z| z * (1f32 - z));

        sigmoid
    }
}
