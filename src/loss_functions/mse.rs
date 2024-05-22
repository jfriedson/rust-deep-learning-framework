use crate::loss_functions::loss_function::LossFunction;
use ndarray::{ArrayD, ArrayViewD};

pub struct MSE {}

#[allow(unused)]
impl MSE {
    pub fn new() -> Self {
        MSE {}
    }
}

impl LossFunction for MSE {
    fn forward(&self, predictions: &ArrayViewD<f32>, truths: &ArrayViewD<f32>) -> f32 {
        let mut square_error = predictions - truths;
        square_error.mapv_inplace(|x| x.powi(2));

        square_error.mean().unwrap()
    }

    fn backward(&self, predictions: &ArrayViewD<f32>, truths: &ArrayViewD<f32>) -> ArrayD<f32> {
        predictions - truths
    }
}
