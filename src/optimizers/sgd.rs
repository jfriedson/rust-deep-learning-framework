use crate::neural_network::model::Model;
use crate::optimizers::optimizer::Optimizer;
use ndarray::{ArrayViewD, Axis, IxDyn};
use ndarray::iter::AxisIter;
use rand::seq::IteratorRandom;

pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        SGD { learning_rate }
    }
}

impl<'a> Optimizer<'a> for SGD {
    fn prepare(&self, model: &mut Model, training_data_dim: IxDyn) {
        let mut next_input_dim = training_data_dim.clone();

        for module in model.modules.iter_mut() {
            next_input_dim = module.prepare(1, next_input_dim);
        }
    }

    fn data_batch(&'a mut self, training_data_iter: &'a mut AxisIter<'a, f32, IxDyn>) -> Option<ArrayViewD<f32>> {
        training_data_iter.choose(&mut rand::thread_rng())
    }
}

// impl SGD {
//     fn gradient_adjustment(
//         &self,
//         mut weights: ArrayViewMutD<f32>,
//         mut biases: ArrayViewMutD<f32>,
//         mut gradients: ArrayViewMutD<f32>,
//     ) {
//         gradients *= self.learning_rate;
//
//         weights *= weights.mul(&gradients);
//         biases *= biases.mul(&gradients);
//     }
// }
