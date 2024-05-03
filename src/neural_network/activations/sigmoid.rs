use crate::neural_network::module::Module;
use ndarray::{Array, ArrayView, IxDyn};

pub struct Sigmoid {}

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid {}
    }
}

impl Module for Sigmoid {
    fn trainable(&self) -> bool {
        false
    }

    fn forward(&self, z: ArrayView<f32, IxDyn>, training: bool) -> Array<f32, IxDyn> {
        z.mapv(|el| 1. / (1. + (-el).exp()))
    }

    // fn backward(&self, z: ArrayBase<S, D>) -> Array<f32, D> {
    //     let mut sigmoid = self.forward(z);
    //
    //     sigmoid.mapv_inplace(|el| el * (1f32 - el));
    //
    //     sigmoid
    // }
}
