use crate::data_loader::data_loader::DataLoader;
use crate::loss_functions::loss_function::LossFunction;
use crate::neural_network::model::Model;
use crate::optimizers::optimizer::Optimizer;
use ndarray::{Array1, ArrayD, Axis, RemoveAxis};
use std::ops::Div;

pub struct ModelTrainer<'a> {
    model: &'a mut Model,
    loss_fn: Box<dyn LossFunction>,
    optimizer: Box<dyn Optimizer>,
}

impl<'a> ModelTrainer<'a> {
    pub fn new(
        model: &'a mut Model,
        loss_fn: Box<dyn LossFunction>,
        optimizer: Box<dyn Optimizer>,
    ) -> Self {
        ModelTrainer {
            model,
            loss_fn,
            optimizer,
        }
    }

    pub fn train(&mut self, data_loader: &mut DataLoader<f32>, epochs: usize) {
        self.optimizer
            .prepare(self.model, data_loader.get_data_dim().remove_axis(Axis(0)));

        for iteration in 0..epochs {
            let mut losses = Vec::<f32>::new();

            // TODO: abstract training data iter within optimizer
            for data_sample in data_loader.rand_iter() {
                let (training_input, output_truth) = data_sample.split_at(Axis(0), 1);

                let output_prediction = self
                    .model
                    .forward(training_input.remove_axis(Axis(0)).into_dyn())
                    .into_dyn();

                let loss = self
                    .loss_fn
                    .forward(&output_prediction.view(), &output_truth);
                losses.push(loss.mean().unwrap());

                let loss_prime = self
                    .loss_fn
                    .backward(&output_prediction.view(), &output_truth);
                self.model.backward(loss_prime.view());

                self.model.apply_gradients(&self.optimizer);
            }

            let loss = losses.iter().sum::<f32>().div(losses.len() as f32);
            println!("iteration: {} loss: {}", iteration + 1, loss);
        }
    }
}
