use crate::neural_network::model::Model;
use ndarray::{ArrayViewD, IxDyn};
use ndarray::iter::AxisIter;

pub trait Optimizer<'a> {
    fn prepare(&self, model: &mut Model, training_data_dim: IxDyn);

    fn data_batch(&'a mut self, training_data_iter: &'a mut AxisIter<'a, f32, IxDyn>) -> Option<ArrayViewD<f32>>;
}
