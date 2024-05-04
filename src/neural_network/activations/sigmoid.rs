use crate::neural_network::module::Module;
use ndarray::{Array0, ArrayD, ArrayViewD, IxDyn};

pub struct Sigmoid {
    inputs: ArrayD<f32>,
}

impl Sigmoid {
    pub fn new() -> Self {
        let inputs = Array0::<f32>::into_dyn(Default::default());

        Sigmoid {
            inputs,
        }
    }
}

impl Module for Sigmoid {
    fn infer(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        input.mapv(|z| 1. / (1. + (-z).exp()))
    }

    fn prepare(&self, batch_size: usize, input_dim: IxDyn) -> IxDyn {
        // TODO: set gradient array size

        input_dim
    }

    fn forward(&self, z: ArrayViewD<f32>) -> ArrayD<f32> {
        // TODO: push gradient to array

        self.infer(z)
    }

    fn backward(&self, loss: ArrayViewD<f32>) -> ArrayD<f32> {
        let mut sigmoid = self.forward(loss);

        //sigmoid.mul_assign(1. - sigmoid);
        sigmoid.mapv_inplace(|el| el * (1. - el));

        sigmoid
    }
}
