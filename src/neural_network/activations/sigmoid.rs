use ndarray::{Array, ArrayBase, Data, Dimension};
use crate::neural_network::activations::activation::Activation;
use crate::neural_network::neural_module::NeuralModule;

pub struct Sigmoid {
}

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid {
        }
    }
}

impl NeuralModule for Sigmoid {
    fn trainable(&self) -> bool {
        false
    }
}

impl<S, D> Activation<S, D> for Sigmoid
where
    S: Data<Elem = f32>,
    D: Dimension,
{
    fn forward(&self, z: ArrayBase<S, D>) -> Array<f32, D> {
        z.mapv(|el| 1f32/(1f32 + (-el).exp()))
    }
    fn backward(&self, z: ArrayBase<S, D>) -> Array<f32, D> {
        let mut sigmoid = self.forward(z);

        sigmoid.mapv_inplace(|el| el * (1f32 - el));

        sigmoid
    }
}
