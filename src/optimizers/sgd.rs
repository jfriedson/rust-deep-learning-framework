use crate::neural_network::model::Model;
use crate::optimizers::optimizer::Optimizer;
use ndarray::{ArrayViewMutD, IxDyn};
use std::ops::MulAssign;

pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        SGD { learning_rate }
    }
}

impl Optimizer for SGD {
    fn prepare(&self, model: &mut Model, training_data_dim: IxDyn) {
        let mut next_input_dim = training_data_dim;

        for module in model.modules.iter_mut() {
            next_input_dim = module.prepare(next_input_dim);
        }
    }

    fn adjust_gradients(&self, mut gradients_array: ArrayViewMutD<f32>) {
        gradients_array.mul_assign(self.learning_rate)
    }
}
