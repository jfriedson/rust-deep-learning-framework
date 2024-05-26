use crate::data_loader::data_loader::DataLoader;
use crate::loss_functions::bce::BCE;
use crate::neural_network::activations::{leaky_relu::LeakyReLU, sigmoid::Sigmoid};
use crate::neural_network::layers::dense::Dense;
use crate::optimizers::sgd::SGD;
use ndarray::array;
use crate::model::Model;
use crate::model_trainer::ModelTrainer;

mod data_loader;
mod loss_functions;
mod neural_network;
mod optimizers;
pub mod model;
pub mod model_trainer;

fn main() {
    let mut neural_net = Model::new();
    let loss_fn = Box::new(BCE::new());
    let optimizer = Box::new(SGD::new(2e-1, Some(1e-6)));

    let mut data_loader = DataLoader::<f32>::from_arrays(
        // input
        array![
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.],
        ]
        .into_dyn(),
        // output
        array![
            [[1., 0.], [0., 0.]],
            [[0., 1.], [0., 0.]],
            [[0., 0.], [1., 0.]],
            [[0., 0.], [0., 1.]],
        ]
        .into_dyn(),
    );

    let mut trainer = ModelTrainer::new(&mut neural_net, loss_fn, optimizer);
    trainer.train(&mut data_loader, 9999);

    for sample in data_loader.iter() {
        let input = sample.0;
        let truth = sample.1;

        let output = neural_net.infer(input.view().into_dyn());

        println!("input: {} output: {} expected: {}", input, output, truth);
    }
}
