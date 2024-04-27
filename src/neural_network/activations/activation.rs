use ndarray::{Array1, ArrayView1};

pub trait Activation {
    fn forward(&self, z: &ArrayView1<f32>) -> Array1<f32>;
    fn backward(&self, z: &ArrayView1<f32>) -> Array1<f32>;
}
