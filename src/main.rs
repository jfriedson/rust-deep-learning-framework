use crate::data_loader::data_loader::DataLoader;
use crate::model::Model;
use crate::model_trainer::ModelTrainer;
use crate::optimizers::sgd::SGD;
use ndarray::array;
use crate::loss_functions::scce::SCCE;

mod data_loader;
mod loss_functions;
pub mod model;
pub mod model_trainer;
mod neural_network;
mod optimizers;

fn main() {
    let mut neural_net = Model::new();
    let loss_fn = Box::new(SCCE::new(1e-6));

    let optimizer = Box::new(SGD::new(1., Some(1e-6)));

    let inputs = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]].into_dyn();
    let outputs = array![
        0.,
        1.,
        2.,
        3.,
    ]
    .into_dyn();

    let mut data_loader = DataLoader::<f32>::from_arrays(inputs, outputs);

    let mut trainer = ModelTrainer::new(&mut neural_net, loss_fn, optimizer);
    trainer.train(&mut data_loader, 9999);

    for sample in data_loader.iter() {
        let input = sample.0;
        let truth = sample.1;

        let output = neural_net.infer(input.view().into_dyn());

        println!("input: {} output: {} expected: {}", input, output, truth);
    }
}
