use std::cell::RefCell;
use ndarray::ArrayViewD;
use crate::loss_functions::loss_function::LossFunction;
use crate::neural_network::model::Model;
use crate::optimizers::optimizer::Optimizer;

pub struct ModelTrainer<'a> {
    model: &'a Model,
    loss_fn: Box<dyn LossFunction>,
    optimizer: Box<dyn Optimizer>,
}

impl<'a> ModelTrainer<'a> {
    pub fn new(model: &'a Model, loss_fn: Box<dyn LossFunction>, optimizer: Box<dyn Optimizer>) -> Self {
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
            let loss = self.optimizer.step(&training_data);

            println!("iteration: {} loss: {}", iteration + 1, loss);
        }
    }
}
