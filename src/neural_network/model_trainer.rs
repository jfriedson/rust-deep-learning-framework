use crate::data_loader::data_loader::DataLoader;
use crate::loss_functions::loss_function::LossFunction;
use crate::neural_network::model::Model;
use crate::optimizers::optimizer::Optimizer;
use ndarray::{Axis, RemoveAxis};
use std::ops::Div;

pub struct ModelTrainer<'a> {
    model: &'a mut Model,
    loss_fn: Box<dyn LossFunction>,
    optimizer: Box<dyn Optimizer>,
}

#[allow(unused)]
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
                let training_input = data_sample.index_axis(Axis(0), 0).into_dyn();
                let output_truth = data_sample.index_axis(Axis(0), 1).into_dyn();

                let output_prediction = self.model.forward(training_input).into_dyn();

                let loss = self
                    .loss_fn
                    .forward(&output_prediction.view(), &output_truth);
                losses.push(loss);

                let loss_prime = self
                    .loss_fn
                    .backward(&output_prediction.view(), &output_truth);
                self.model.backward(loss_prime.view());

                self.model.apply_gradients(&self.optimizer);

                self.model.zero_gradients();
            }

            let loss = losses.iter().sum::<f32>().div(losses.len() as f32);
            println!("iteration: {} loss: {}", iteration + 1, loss);
        }
    }
}
