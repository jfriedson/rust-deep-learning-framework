use ndarray::{array, Axis};
use crate::model::Model;

mod model;
mod loss_functions;
mod neural_network;
mod optimizers;

fn main() {
    let neural_net = Model::new();

    // TODO: implement a data loader
    let training_data = array![
        [[0f32, 0f32], [0f32, 0f32]],
        [[0f32, 1f32], [1f32, 0f32]],
        [[1f32, 0f32], [0f32, 1f32]],
        [[1f32, 1f32], [1f32, 1f32]],
    ];

    //neural_net.train(&training_data.view(), 5);

    for sample in training_data.axis_iter(Axis(0)) {
        let input = sample.row(0);
        let output = neural_net.infer(&input.view());
        println!("{:?}", output);
    }
}
