use crate::loss_functions::mse::MSE;
use crate::neural_network::activations::leaky_relu::LeakyRelu;
use crate::neural_network::activations::sigmoid::Sigmoid;
use crate::neural_network::layers::dense::Dense;
use crate::neural_network::model_builder::ModelBuilder;
use crate::neural_network::model_trainer::ModelTrainer;
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
        .build();
    let loss_fn = Box::new(MSE::new());
    let optimizer = Box::new(SGD::new(1e-3));

    // TODO: implement a data loader
    let training_data = array![
        // input    output
        [[0., 0.], [0., 0.]],
        [[0., 1.], [1., 0.]],
        [[1., 0.], [0., 1.]],
        [[1., 1.], [1., 1.]],
    ];

    let mut trainer = ModelTrainer::new(Box::new(neural_net), loss_fn, optimizer);
    trainer.train(&training_data.view().into_dyn(), 5);

    for sample in training_data.axis_iter(Axis(0)) {
        let input = sample.row(0);

        let output = neural_net.infer(input.into_dyn());

        println!("{:?}", output);
    }
}
