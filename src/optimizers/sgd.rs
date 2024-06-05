use crate::optimizers::optimizer::Optimizer;
use ndarray::{ArrayViewD, ArrayViewMutD};

pub struct SGD {
    learning_rate: f32,
    momentum: f32,
    weight_decay: f32,
}

#[allow(unused)]
impl SGD {
    pub fn new(learning_rate: f32, momentum: Option<f32>, weight_decay: Option<f32>) -> Self {
        SGD {
            learning_rate,
            momentum: momentum.unwrap_or(0.),
            weight_decay: weight_decay.unwrap_or(0.),
        }
    }
}

impl Optimizer for SGD {
    fn adjust_gradients(&self, mut gradients: ArrayViewMutD<f32>, mut gradient_moments: ArrayViewMutD<f32>) {
        gradients *= self.learning_rate;

        if self.momentum != 0. {
            if gradient_moments.shape() != gradients.shape() {
                gradient_moments.assign(&gradients);
            }
            else {
                gradient_moments *= self.momentum;
                gradient_moments += &gradients;
            }

            gradients.assign(&gradient_moments);
        }
    }

    fn adjust_weight_deltas(
        &self,
        mut weight_deltas: ArrayViewMutD<f32>,
        weights: ArrayViewD<f32>,
    ) {
        if self.weight_decay != 0. {
            weight_deltas += &(self.learning_rate * self.weight_decay * &weights);
        }
    }
}
