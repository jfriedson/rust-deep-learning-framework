use crate::optimizers::optimizer::Optimizer;
use ndarray::{ArrayViewD, ArrayViewMutD, Axis, IxDyn};
use std::ops::{Div, Mul, SubAssign};
use crate::neural_network::model::Model;
use crate::neural_network::module::Module;

pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        SGD {
            learning_rate,
        }
    }
}

impl Optimizer for SGD {
    fn prepare(&self, modules: &mut Vec<Box<dyn Module>>, mut training_data_dim: IxDyn) {
        for module in modules.iter_mut() {
            training_data_dim = module.prepare(1, training_data_dim);
        }
    }

    fn step(&self, model: &Model, training_data: &ArrayViewD<f32>) -> f32 {
        let mut losses = Vec::<f32>::new();

        for training_sample in training_data.axis_iter(Axis(0)) {
            let (training_input, mut output_truth) = training_sample.split_at(Axis(0), 1);

            let output_prediction = model.forward(training_input.remove_axis(Axis(0)).into_dyn());

            let loss = model
                .loss_fn
                .forward(output_prediction.view().into_dyn(), output_truth.remove_axis(Axis(0)).into_dyn());
            losses.push(loss.mean().unwrap());

            model.backward(loss.view());

            // model.apply_gradients(loss.view(), self.gradient_adjustment);
        }

        losses.iter().sum::<f32>().div(losses.len() as f32)
    }
}

impl SGD {
    fn gradient_adjustment(
        &self,
        mut weights: ArrayViewMutD<f32>,
        mut biases: ArrayViewMutD<f32>,
        gradients: ArrayViewD<f32>,
    ) {
        let adjusted_gradients = &gradients * self.learning_rate;

        weights.sub_assign(&weights.mul(&adjusted_gradients));
        biases.sub_assign(&biases.mul(&adjusted_gradients));
    }
}