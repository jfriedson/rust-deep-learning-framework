use crate::neural_network::model::Model;
use ndarray::{ArrayViewMutD, IxDyn};

pub trait Optimizer {
    fn prepare(&self, model: &mut Model, training_data_dim: IxDyn);

    fn adjust_gradients(&self, gradients_array: ArrayViewMutD<f32>);
}
