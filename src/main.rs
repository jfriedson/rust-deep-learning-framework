use crate::data_loader::data_loader::DataLoader;
use crate::loss_functions::mse::MSE;
use crate::neural_network::activations::leaky_relu::LeakyRelu;
use crate::neural_network::activations::sigmoid::Sigmoid;
use crate::neural_network::layers::dense::Dense;
use crate::neural_network::model_builder::ModelBuilder;
use crate::neural_network::model_trainer::ModelTrainer;
use crate::optimizers::sgd::SGD;
use ndarray::{array, Axis};

mod data_loader;
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
        //.add_module(Box::new(Sigmoid::new()))
        .build();
    let loss_fn = Box::new(MSE::new());
    let optimizer = Box::new(SGD::new(1e-2, Some(1e-6)));

    let mut data_loader = DataLoader::<f32>::from_array(
        array![
            // input    output
            [[0., 0.], [0., 0.]],
            [[0., 1.], [1., 0.]],
            [[1., 0.], [0., 1.]],
            [[1., 1.], [1., 1.]],
        ]
        .into_dyn(),
    );

    let mut trainer = ModelTrainer::new(&mut neural_net, loss_fn, optimizer);
    trainer.train(&mut data_loader, 9999);

    for sample in data_loader.iter() {
        let input = sample.index_axis(Axis(0), 0);
        let truth = sample.index_axis(Axis(0), 1);

        let output = neural_net.infer(input.view().into_dyn());

        println!("input: {} output: {} expected: {}", input, output, truth);
    }
}
