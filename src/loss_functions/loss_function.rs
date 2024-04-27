use ndarray::{Array1, ArrayView1};

pub trait LossFunction {
    fn forward(&self, predictions: &ArrayView1<f32>, truths: &ArrayView1<f32>) -> Array1<f32>;
    fn backward(&self, predictions: &ArrayView1<f32>, truths: &ArrayView1<f32>) -> Array1<f32>;
}
