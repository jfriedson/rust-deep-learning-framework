use crate::data_loader::data_loader::DataLoader;
use crate::loss_functions::mse::MSE;
use crate::neural_network::activations::leaky_relu::LeakyRelu;
use crate::neural_network::activations::sigmoid::Sigmoid;
use crate::neural_network::layers::dense::Dense;
use crate::neural_network::model_builder::ModelBuilder;
use crate::neural_network::model_trainer::ModelTrainer;
use crate::optimizers::sgd::SGD;
use ndarray::array;

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
    let mut data_loader = DataLoader::<f32>::from_array(training_data.into_dyn());

    let mut trainer = ModelTrainer::new(&mut neural_net, loss_fn, optimizer);
    trainer.train(&mut data_loader, 5);

    for sample in data_loader.iter() {
        let input = sample.rows().into_iter().next().unwrap();

        let output = neural_net.infer(input.into_dyn());

        println!("{:?}", output);
    }
}
