use crate::neural_network::model::Model;
use ndarray::IxDyn;

pub trait Optimizer {
    fn prepare(&self, model: &mut Model, training_data_dim: IxDyn);

    fn batch_size(&mut self) -> f32;
}
