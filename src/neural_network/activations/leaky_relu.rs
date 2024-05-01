use ndarray::{Array, ArrayView, IxDyn};
use crate::neural_network::module::Module;

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

impl Module for LeakyRelu {
    fn trainable(&self) -> bool {
        false
    }

    fn forward(&self, z: ArrayView<f32, IxDyn>) -> Array<f32, IxDyn> {
        z.mapv(|el| f32::max(el, el * self.negative_slope))
    }

    // fn backward(&self, z: ArrayBase<S, D>) -> Array<f32, D>
    // where
    //     S: Data<Elem = f32>,
    //     D: Dimension,
    // {
    //     z.mapv(|el| if el > 0f32 { 1f32 } else { self.negative_slope })
    // }
}
