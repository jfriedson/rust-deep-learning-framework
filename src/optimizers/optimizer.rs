use ndarray::{ArrayViewD, IxDyn};
use crate::neural_network::model::Model;
use crate::neural_network::module::Module;

pub trait Optimizer {
    fn prepare(&self, modules: &mut Vec<Box<dyn Module>>, training_data_dim: IxDyn);

    fn step(&self, model: &Model, training_data: &ArrayViewD<f32>) -> f32;
}
