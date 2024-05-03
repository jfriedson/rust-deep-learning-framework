use crate::optimizers::optimizer::Optimizer;
use ndarray::{ArrayView, ArrayView3, ArrayViewMut, Axis, IxDyn};
use std::ops::{Div, Mul, SubAssign};
use crate::neural_network::model::Model;

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
    fn step(&self, model: &Model, training_data: ArrayView3<f32>) -> f32 {
        let mut losses = Vec::<f32>::new();

        for training_sample in training_data.axis_iter(Axis(0)) {
            let training_input = training_sample.row(0);
            let output_truth = training_sample.row(1);

            let output_prediction = model.forward(training_input.into_dyn(), true);

            let loss = model
                .loss_fn
                .forward(output_prediction.view().into_dyn(), output_truth.into_dyn());
            losses.push(loss.mean().unwrap());

            // TODO: backprop
            self.backpropagate(model);

            // TODO: optimizer
            //self.apply_gradients()
        }

        losses.iter().sum::<f32>().div(losses.len() as f32)
    }
}

impl SGD {
    fn backpropagate(&self, model: &Model) {
        for module in model.modules {

        }
    }

    fn apply_gradients(
        &self,
        mut weights: ArrayViewMut<f32, IxDyn>,
        mut biases: ArrayViewMut<f32, IxDyn>,
        gradients: ArrayView<f32, IxDyn>,
    ) {
        let adjusted_gradients = &gradients * self.learning_rate;

        weights.sub_assign(weights.mul(&adjusted_gradients));
        biases.sub_assign(biases.mul(&adjusted_gradients));
    }
}