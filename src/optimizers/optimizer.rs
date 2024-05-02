use ndarray::{ArrayView, ArrayViewMut, IxDyn};

pub trait Optimizer {
    fn apply_gradients(
        &self,
        weights: &mut ArrayViewMut<f32, IxDyn>,
        biases: &mut ArrayViewMut<f32, IxDyn>,
        gradients: &ArrayView<f32, IxDyn>,
    ) {
    }
}
