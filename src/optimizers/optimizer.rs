use crate::neural_network::model::Model;
use ndarray::{ArrayViewD, ArrayViewMutD, IxDyn};

pub trait Optimizer {
    fn prepare(&self, model: &mut Model, training_data_dim: IxDyn);

    fn adjust_gradients(&self, gradients: ArrayViewMutD<f32>);

    fn adjust_weight_deltas(&self, weight_deltas: ArrayViewMutD<f32>, weights: ArrayViewD<f32>);
}
