use crate::loss_functions::loss_function::LossFunction;
use ndarray::{ArrayD, ArrayViewD};

pub struct CCE {
    epsilon: f32,
}

#[allow(unused)]
impl CCE {
    pub fn new(epsilon: f32) -> Self {
        CCE {
            epsilon
        }
    }
}

impl LossFunction for CCE {
    fn forward(&self, predictions: &ArrayViewD<f32>, truths: &ArrayViewD<f32>) -> f32 {
        let pred_log = predictions.mapv(|x| (x + self.epsilon).ln());

        -(truths * pred_log).sum()
    }

    fn backward(&self, predictions: &ArrayViewD<f32>, truths: &ArrayViewD<f32>) -> ArrayD<f32> {
        -truths/(predictions + self.epsilon)
    }
}
