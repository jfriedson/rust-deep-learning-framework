use crate::loss_functions::loss_function::LossFunction;
use ndarray::{ArrayD, ArrayViewD};
use ndarray_rand::rand_distr::num_traits::abs;

pub struct MAE {}

#[allow(unused)]
impl MAE {
    pub fn new() -> Self {
        MAE {}
    }
}

impl LossFunction for MAE {
    fn forward(&self, predictions: &ArrayViewD<f32>, truths: &ArrayViewD<f32>) -> f32 {
        let mut absolute_error = predictions - truths;
        absolute_error.mapv_inplace(|x| abs(x));

        absolute_error.mean().unwrap()
    }

    fn backward(&self, predictions: &ArrayViewD<f32>, truths: &ArrayViewD<f32>) -> ArrayD<f32> {
        ArrayD::<f32>::from_shape_fn(predictions.raw_dim(), |i| {
            if predictions[&i] > truths[&i] {
                1.
            } else if predictions[&i] < truths[&i] {
                -1.
            } else {
                0.
            }
        })
    }
}
