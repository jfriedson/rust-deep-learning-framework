use ndarray::{Array1, ArrayView1};

pub trait Layer {
    fn new(input_count: usize, output_count: usize) -> Self;
    fn infer(&self, input: &ArrayView1<f32>) -> Array1<f32>;
}
