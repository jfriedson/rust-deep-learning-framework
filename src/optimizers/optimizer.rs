use ndarray::{ArrayViewD, IxDyn};
use crate::neural_network::model::Model;

pub trait Optimizer {
    fn prepare(&self, model: &Model, training_data: IxDyn);

    fn step(&self, model: &Model, training_data: &ArrayViewD<f32>) -> f32;
}
