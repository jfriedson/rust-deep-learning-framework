use ndarray::{Array, ArrayView, IxDyn};
use crate::neural_network::module::Module;

pub struct Sigmoid {
}

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid {
        }
    }
}

impl Module for Sigmoid {
    fn trainable(&self) -> bool {
        false
    }

    fn forward(&self, z: ArrayView<f32, IxDyn>) -> Array<f32, IxDyn> {
        z.mapv(|el| 1f32/(1f32 + (-el).exp()))
    }

    // fn backward(&self, z: ArrayBase<S, D>) -> Array<f32, D> {
    //     let mut sigmoid = self.forward(z);
    //
    //     sigmoid.mapv_inplace(|el| el * (1f32 - el));
    //
    //     sigmoid
    // }
}
