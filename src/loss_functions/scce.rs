use crate::loss_functions::loss_function::LossFunction;
use ndarray::{Array1, ArrayD, ArrayViewD, Dimension, Ix0};

pub struct SCCE {
    epsilon: f32,
}

#[allow(unused)]
impl SCCE {
    pub fn new(epsilon: f32) -> Self {
        SCCE {
            epsilon,
        }
    }
}

impl LossFunction for SCCE {
    fn forward(&self, predictions: &ArrayViewD<f32>, truths: &ArrayViewD<f32>) -> f32 {
        let category = *truths.view().into_dimensionality::<Ix0>().unwrap().into_scalar();

        -(predictions[category as usize] + self.epsilon).ln()
    }

    fn backward(&self, predictions: &ArrayViewD<f32>, truths: &ArrayViewD<f32>) -> ArrayD<f32> {
        let mut truths_one_hot = Array1::<f32>::zeros(predictions.raw_dim().size());

        let category = *truths.view().into_dimensionality::<Ix0>().unwrap().into_scalar();
        truths_one_hot[category as usize] = 1.;

        -truths_one_hot/(predictions + self.epsilon)
    }
}
