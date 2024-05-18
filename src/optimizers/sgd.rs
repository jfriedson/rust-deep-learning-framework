use crate::neural_network::model::Model;
use crate::optimizers::optimizer::Optimizer;
use ndarray::{ArrayViewD, ArrayViewMutD, IxDyn};

pub struct SGD {
    learning_rate: f32,
    weight_decay: f32,
}

impl SGD {
    pub fn new(learning_rate: f32, weight_decay: Option<f32>) -> Self {
        SGD {
            learning_rate,
            weight_decay: weight_decay.unwrap_or(0.),
        }
    }
}

impl Optimizer for SGD {
    fn prepare(&self, model: &mut Model, training_data_dim: IxDyn) {
        let mut next_input_dim = training_data_dim;

        for module in model.modules.iter_mut() {
            next_input_dim = module.prepare(next_input_dim);
        }
    }

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
