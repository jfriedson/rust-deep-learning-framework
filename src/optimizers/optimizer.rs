use ndarray::{ArrayView, ArrayView3, ArrayViewMut, IxDyn};
use crate::neural_network::model::Model;

pub trait Optimizer {
    fn step(&self, model: &Model, training_data: ArrayView3<f32>) -> f32;
}
