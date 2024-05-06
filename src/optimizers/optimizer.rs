use std::cell::RefCell;
use crate::neural_network::model::Model;
use crate::neural_network::module::Module;
use ndarray::{ArrayViewD, IxDyn};

pub trait Optimizer {
    fn prepare(&self, model: &mut Model, training_data_dim: IxDyn);

    fn step(&mut self, training_data: &ArrayViewD<f32>) -> f32;
}
