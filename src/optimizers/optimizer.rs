use ndarray::{ArrayViewD, ArrayViewMutD};

pub trait Optimizer {
    fn adjust_gradients(&self, gradients: ArrayViewMutD<f32>, gradient_velocities: ArrayViewMutD<f32>);

    fn adjust_weight_deltas(&self, weight_deltas: ArrayViewMutD<f32>, weights: ArrayViewD<f32>);
}
