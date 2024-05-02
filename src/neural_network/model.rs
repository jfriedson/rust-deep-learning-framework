use crate::loss_functions::loss_function::LossFunction;
use crate::neural_network::model_builder::ModelBuilder;
use crate::neural_network::module::Module;
use crate::optimizers::optimizer::Optimizer;
use ndarray::{Array, ArrayView, ArrayView3, Axis, IxDyn};
use std::ops::Div;

pub struct Model {
    pub(crate) modules: Vec<Box<dyn Module>>,
    pub(crate) loss_fn: Box<dyn LossFunction>,
    pub(crate) optimizer: Box<dyn Optimizer>,
}

impl Model {
    pub fn builder() -> ModelBuilder {
        ModelBuilder::new()
    }

    pub fn infer(&self, input: ArrayView<f32, IxDyn>) -> Array<f32, IxDyn> {
        let mut next_input = input.to_owned();

        for module in self.modules.iter() {
            next_input = module.forward(next_input.view());
        }

        next_input
    }

    pub fn train(&self, training_data: &ArrayView3<f32>, epochs: u32) {
        for iteration in 0..epochs {
            let mut losses = Vec::<f32>::new();

            for training_sample in training_data.axis_iter(Axis(0)) {
                let training_input = training_sample.row(0);
                let output_truth = training_sample.row(1);

                let output_prediction = self.infer(training_input.into_dyn());

                let loss = self
                    .loss_fn
                    .forward(output_prediction.view().into_dyn(), output_truth.into_dyn());
                losses.push(loss.mean().unwrap());

                // TODO: backprop
                // TODO: optimizer
            }

            let mse = losses.iter().sum::<f32>().div(losses.len() as f32);
            println!("iteration: {} mse: {}", iteration + 1, mse);
        }
    }
}
