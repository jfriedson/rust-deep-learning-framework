use std::ops::Div;
use ndarray::{Array1, ArrayView1, ArrayView3, Axis};
use crate::loss_functions::loss_function::LossFunction;
use crate::loss_functions::mse::MSE;
use crate::neural_network::activations::activation::Activation;
use crate::neural_network::activations::leaky_relu::LeakyRelu;
use crate::neural_network::activations::sigmoid::Sigmoid;
use crate::neural_network::layers::layer::Layer;
use crate::neural_network::layers::dense::Dense;

pub struct Model {
    hidden_layer_1: Dense,
    hidden_layer_2: Dense,
    leaky_relu: LeakyRelu,
    output_layer: Dense,
    sigmoid: Sigmoid,
    loss_fn: MSE,
}

impl Model {
    pub fn new() -> Self {
        let hidden_layer_1 = Dense::new(2, 4);
        let hidden_layer_2 = Dense::new(4, 4);
        let leaky_relu = LeakyRelu::new(0.1f32);
        let output_layer = Dense::new(4, 2);
        let sigmoid = Sigmoid::new();

        let loss_fn = MSE::new();

        Model {
            hidden_layer_1,
            hidden_layer_2,
            leaky_relu,
            output_layer,
            sigmoid,
            loss_fn
        }
    }

    pub fn infer(&self, input: ArrayView1<f32>) -> Array1<f32> {
        let mut next_input;

        next_input = self.hidden_layer_1.infer(input);
        next_input = self.leaky_relu.forward(next_input);

        next_input = self.hidden_layer_2.infer(next_input);
        next_input = self.leaky_relu.forward(next_input);

        next_input = self.output_layer.infer(next_input);
        self.sigmoid.forward(next_input)
    }

    pub fn train(&self, training_data: &ArrayView3<f32>, epochs: u32) {
        for iteration in 0..epochs {
            let mut losses = Vec::<f32>::new();

            for training_sample in training_data.axis_iter(Axis(0)) {
                let training_input = training_sample.row(0);
                let output_truth = training_sample.row(1);

                let output_prediction = self.infer(training_input);

                let loss = self.loss_fn.forward(output_prediction, output_truth.to_owned());
                losses.push(loss.mean().unwrap());

                // backprop
            }

            let mse = losses.iter().sum::<f32>().div(losses.len() as f32);
            println!("iteration: {} mse: {}", iteration + 1, mse);
        }
    }
}
