use ndarray::{Array, ArrayBase, Data, Dimension};
use crate::neural_network::neural_module::NeuralModule;

pub trait Activation<S, D>: NeuralModule
where
    S: Data<Elem = f32>,
    D: Dimension,
{
    fn forward(&self, z: ArrayBase<S, D>) -> Array<f32, D>;

    fn backward(&self, z: ArrayBase<S, D>) -> Array<f32, D>;
}
