use crate::neural_network::model::Model;
use crate::optimizers::optimizer::Optimizer;
use ndarray::{ArrayViewD, Axis, IxDyn};
use ndarray::iter::AxisIter;
use rand::seq::IteratorRandom;

pub struct SGD<'a> {
    learning_rate: f32,

    data_iter: Option<AxisIter<'a, f32, IxDyn>>,
}

impl<'a> SGD<'a> {
    pub fn new(learning_rate: f32) -> Self {
        let data_iter = None;

        SGD { learning_rate, data_iter }
    }
}

impl<'a> Optimizer<'a> for SGD<'a> {
    fn prepare(&self, model: &mut Model, training_data_dim: IxDyn) {
        let mut next_input_dim = training_data_dim.clone();

        for module in model.modules.iter_mut() {
            next_input_dim = module.prepare(1, next_input_dim);
        }
    }

    fn data_batch(&mut self, training_data: &'a ArrayViewD<f32>) -> Option<ArrayViewD<f32>> {
        if self.data_iter.is_none() {
            self.data_iter = Some(training_data.axis_iter(Axis(0)));
        }

        self.data_iter.as_ref().unwrap().clone().choose(&mut rand::thread_rng())
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
