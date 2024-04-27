use ndarray::{Array1, ArrayView1};
use crate::loss_functions::loss_function::LossFunction;

pub struct MSE {
}

impl MSE {
    pub fn new() -> Self {
        MSE {
        }
    }
}

impl LossFunction for MSE {
    fn forward(&self, predictions: &ArrayView1<f32>, truths: &ArrayView1<f32>) -> Array1<f32> {
        let mut mse = predictions - truths;
        mse.mapv_inplace(|x| x.powi(2));

        mse
    }

    fn backward(&self, predictions: &ArrayView1<f32>, truths: &ArrayView1<f32>) -> Array1<f32> {
        predictions - truths
    }
}
