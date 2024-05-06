use std::ops::Div;
use ndarray::{ArrayViewD, Axis};
use crate::loss_functions::loss_function::LossFunction;
use crate::neural_network::model::Model;
use crate::optimizers::optimizer::Optimizer;

pub struct ModelTrainer<'a> {
    model: &'a mut Model,
    loss_fn: Box<dyn LossFunction>,
    optimizer: Box<dyn Optimizer>,
}

impl<'a> ModelTrainer<'a> {
    pub fn new(model: &'a mut Model, loss_fn: Box<dyn LossFunction>, optimizer: Box<dyn Optimizer>) -> Self {
        ModelTrainer {
            model,
            loss_fn,
            optimizer,
        }
    }

    pub fn train(&mut self, training_data: ArrayViewD<f32>, epochs: usize) {
        self.optimizer
            .prepare(self.model, training_data.raw_dim());

        for iteration in 0..epochs {
            let mut losses = Vec::<f32>::new();

            for step in self.optimizer {
                let (training_input, output_truth) = training_sample.split_at(Axis(0), 1);
                let output_truth = output_truth.remove_axis(Axis(0)).into_dyn();

                let output_prediction = ref_model.forward(training_input.remove_axis(Axis(0)).into_dyn());

                let loss = ref_model.loss_fn.forward(
                    &output_prediction.view().into_dyn(),
                    &output_truth,
                );
                losses.push(loss.mean().unwrap());

                let loss_prime = ref_model.loss_fn.backward(
                    &output_prediction.view().into_dyn(),
                    &output_truth,
                );
                ref_model.backward(loss_prime.view());

                // model.apply_gradients(loss.view(), self.gradient_adjustment);
            }

            losses.iter().sum::<f32>().div(losses.len() as f32)
        }

            println!("iteration: {} loss: {}", iteration + 1, loss);
        }
    }
}
