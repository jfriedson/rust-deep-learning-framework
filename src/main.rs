use crate::loss_functions::mse::MSE;
use crate::neural_network::activations::leaky_relu::LeakyRelu;
use crate::neural_network::activations::sigmoid::Sigmoid;
use crate::neural_network::layers::dense::Dense;
use crate::neural_network::model_builder::ModelBuilder;
use crate::optimizers::sgd::SGD;
use ndarray::{array, Axis};

mod loss_functions;
mod neural_network;
mod optimizers;

fn main() {
    let mut neural_net = ModelBuilder::new()
        .add_module(Box::new(Dense::new(2, 4)))
        .add_module(Box::new(LeakyRelu::new(0.1)))
        .add_module(Box::new(Dense::new(4, 4)))
        .add_module(Box::new(LeakyRelu::new(0.1)))
        .add_module(Box::new(Dense::new(4, 2)))
        .add_module(Box::new(Sigmoid::new()))
        .set_loss_fn(Box::new(MSE::new()))
        .set_optimizer(Box::new(SGD::new(1e-3)))
        .build();

    // TODO: implement a data loader
    let training_data = array![
        // input    output
        [[0., 0.], [0., 0.]],
        [[0., 1.], [1., 0.]],
        [[1., 0.], [0., 1.]],
        [[1., 1.], [1., 1.]],
    ];

    neural_net.train(training_data.view().into_dyn(), 5);

    for sample in training_data.axis_iter(Axis(0)) {
        let input = sample.row(0);

        let output = neural_net.forward(input.into_dyn());

        println!("{:?}", output);
    }
}
