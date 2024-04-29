use ndarray::{Array, ArrayBase, Data, Dimension};
use crate::neural_network::activations::activation::Activation;
use crate::neural_network::neural_module::NeuralModule;

pub struct LeakyRelu {
    negative_slope: f32,
}

impl LeakyRelu {
    pub fn new(negative_slope: f32) -> Self {
        LeakyRelu {
            negative_slope
        }
    }
}

impl NeuralModule for LeakyRelu {
    fn trainable(&self) -> bool {
        false
    }
}

impl<S, D> Activation<S, D> for LeakyRelu
where
    S: Data<Elem = f32>,
    D: Dimension,
{
    fn forward(&self, z: ArrayBase<S, D>) -> Array<f32, D>
    where
        S: Data<Elem = f32>,
        D: Dimension,
    {
        z.mapv(|el| f32::max(el, el * self.negative_slope))
    }

    fn backward(&self, z: ArrayBase<S, D>) -> Array<f32, D>
    where
        S: Data<Elem = f32>,
        D: Dimension,
    {
        z.mapv(|el| if el > 0f32 { 1f32 } else { self.negative_slope })
    }
}
