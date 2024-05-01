use ndarray::{array, Axis};
use crate::loss_functions::mse::MSE;
use crate::neural_network::activations::leaky_relu::LeakyRelu;
use crate::neural_network::activations::sigmoid::Sigmoid;
use crate::neural_network::layers::dense::Dense;
use crate::neural_network::model_builder::ModelBuilder;

mod loss_functions;
mod neural_network;
mod optimizers;

fn main() {
    let neural_net = ModelBuilder::new()
        .add_module(Box::new(Dense::new(2, 4)))
        .add_module(Box::new(LeakyRelu::new(0.1f32)))
        .add_module(Box::new(Dense::new(4, 4)))
        .add_module(Box::new(LeakyRelu::new(0.1f32)))
        .add_module(Box::new(Dense::new(4, 2)))
        .add_module(Box::new(Sigmoid::new()))
        .set_loss_fn(Box::new(MSE::new())).build();

    // TODO: implement a data loader
    let training_data = array![
        [[0f32, 0f32], [0f32, 0f32]],
        [[0f32, 1f32], [1f32, 0f32]],
        [[1f32, 0f32], [0f32, 1f32]],
        [[1f32, 1f32], [1f32, 1f32]],
    ];

    neural_net.train(&training_data.view(), 5);

    for sample in training_data.axis_iter(Axis(0)) {
        let input = sample.row(0);
        let output = neural_net.infer(input.into_dyn());
        println!("{:?}", output);
    }
}
