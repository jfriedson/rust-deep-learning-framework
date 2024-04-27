use ndarray::{Array1, ArrayView1};
use crate::neural_network::activations::activation::Activation;

pub struct LeakyRelu {
    negative_slope: f32,
}

impl LeakyRelu {
    pub fn new(negative_slope: f32) -> Self {
        LeakyRelu {
            negative_slope
        }
    }
}

impl Activation for LeakyRelu {
    fn forward(&self, z: &ArrayView1<f32>) -> Array1<f32> {
        z.mapv(|z| f32::max(&self.negative_slope * z, z))
    }
    fn backward(&self, z: &ArrayView1<f32>) -> Array1<f32> {
        z.mapv(|z| if z > 0f32 { 1f32 } else { self.negative_slope })
    }
}
