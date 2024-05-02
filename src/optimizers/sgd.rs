use crate::optimizers::optimizer::Optimizer;
use ndarray::{ArrayView, ArrayViewMut, IxDyn};
use std::ops::{Mul, SubAssign};

pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        SGD { learning_rate }
    }
}

impl Optimizer for SGD {
    fn apply_gradients(
        &self,
        weights: &mut ArrayViewMut<f32, IxDyn>,
        biases: &mut ArrayViewMut<f32, IxDyn>,
        gradients: &ArrayView<f32, IxDyn>,
    ) {
        let adjusted_gradients = gradients * self.learning_rate;

        weights.sub_assign(&weights.mul(&adjusted_gradients));
        biases.sub_assign(&biases.mul(&adjusted_gradients));
    }
}
