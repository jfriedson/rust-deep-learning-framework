use crate::neural_network::model::Model;
use crate::neural_network::module::Module;
use crate::optimizers::optimizer::Optimizer;
use ndarray::{ArrayViewD, Axis, IxDyn};
use std::ops::Div;

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
        let mut next_input_dim = training_data_dim.clone();

        for module in model.modules.iter_mut() {
            next_input_dim = module.prepare(1, next_input_dim);
        }
    }

    // TODO: convert to iterator
    fn step(&self, training_data: &ArrayViewD<f32>) -> f32 {
        for training_sample in training_data.axis_iter(Axis(0)) {
            let (training_input, output_truth) = training_sample.split_at(Axis(0), 1);
            let training_input = training_input.remove_axis(Axis(0)).into_dyn();
            let output_truth = output_truth.remove_axis(Axis(0)).into_dyn();

            let output_prediction = ref_model.forward(training_input.remove_axis(Axis(0)).into_dyn());
        }
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
