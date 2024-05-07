use crate::neural_network::model::Model;
use ndarray::{ArrayViewD, IxDyn};

pub trait Optimizer<'a> {
    fn prepare(&self, model: &mut Model, training_data_dim: IxDyn);

    fn data_batch(&mut self, training_data: &'a ArrayViewD<f32>) -> Option<ArrayViewD<f32>>;
}
