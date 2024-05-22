use crate::loss_functions::loss_function::LossFunction;
use ndarray::{ArrayD, ArrayViewD};

pub struct BCE {}

#[allow(unused)]
impl BCE {
    pub fn new() -> Self {
        BCE {}
    }
}

impl LossFunction for BCE {
    fn forward(&self, predictions: &ArrayViewD<f32>, truths: &ArrayViewD<f32>) -> f32 {
        let log_1_min_predictions = predictions.mapv(|x| (1. - x).ln());
        let log_predictions = predictions.mapv(|x| (x).ln());

        let bce = (1. - truths) * log_1_min_predictions + truths * log_predictions;

        -bce.mean().unwrap()
    }

    fn backward(&self, predictions: &ArrayViewD<f32>, truths: &ArrayViewD<f32>) -> ArrayD<f32> {
        predictions - truths
    }
}
