use crate::data_loader::data_loader::DataLoader;
use crate::loss_functions::loss_function::LossFunction;
use crate::neural_network::model::Model;
use crate::optimizers::optimizer::Optimizer;
use ndarray::{Array1, ArrayD, Axis};
use std::ops::Div;

pub struct ModelTrainer<'a> {
    model: &'a mut Model,
    loss_fn: Box<dyn LossFunction>,
    optimizer: Box<dyn Optimizer>,
    gradients: ArrayD<f32>,
}

impl<'a> ModelTrainer<'a> {
    pub fn new(
        model: &'a mut Model,
        loss_fn: Box<dyn LossFunction>,
        optimizer: Box<dyn Optimizer>,
    ) -> Self {
        let gradients = Array1::<f32>::into_dyn(Default::default());

        print!("{:?}", gradients);
        ModelTrainer {
            model,
            loss_fn,
            optimizer,
            gradients,
        }
    }

    pub fn train(&mut self, data_loader: &mut DataLoader<f32>, epochs: usize) {
        self.optimizer
            .prepare(self.model, data_loader.get_data_dim());

        for iteration in 0..epochs {
            let mut losses = Vec::<f32>::new();

            for data_sample in data_loader.iter() {
                let (training_input, output_truth) = data_sample.split_at(Axis(0), 1);

                let output_prediction = self
                    .model
                    .forward(training_input.remove_axis(Axis(0)).into_dyn());

                let loss = self
                    .loss_fn
                    .forward(&output_prediction.view().into_dyn(), &output_truth);
                losses.push(loss.mean().unwrap());

                let loss_prime = self
                    .loss_fn
                    .backward(&output_prediction.view().into_dyn(), &output_truth);
                self.model.backward(loss_prime.view());

                // TODO: optimizer.optimize_gradients()
                // TODO: model.apply_gradients(loss.view(), self.gradient_adjustment);
            }

            let loss = losses.iter().sum::<f32>().div(losses.len() as f32);
            println!("iteration: {} loss: {}", iteration + 1, loss);
        }
    }
}
