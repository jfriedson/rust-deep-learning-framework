use crate::optimizers::optimizer::Optimizer;
use ndarray::{ArrayViewD, ArrayViewMutD};

pub struct SGD {
    learning_rate: f32,
    weight_decay: f32,
}

#[allow(unused)]
impl SGD {
    pub fn new(learning_rate: f32, weight_decay: Option<f32>) -> Self {
        SGD {
            learning_rate,
            weight_decay: weight_decay.unwrap_or(0.),
        }
    }
}

impl Optimizer for SGD {
    fn adjust_gradients(&self, mut gradients: ArrayViewMutD<f32>) {
        gradients *= self.learning_rate
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
