use crate::neural_network::module::Module;
use ndarray::{Array0, ArrayD, ArrayViewD, Axis, IxDyn};

pub struct Sigmoid {
    gradients: ArrayD<f32>,
}

impl Sigmoid {
    pub fn new() -> Self {
        let gradients = Array0::<f32>::into_dyn(Default::default());

        Sigmoid { gradients }
    }
}

impl Module for Sigmoid {
    fn infer(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        input.mapv(|z| 1. / (1. + (-z).exp()))
    }

    fn prepare(&mut self, batch_size: usize, input_dim: IxDyn) -> IxDyn {
        self.gradients = self
            .gradients
            .clone()
            .into_shape(input_dim.clone())
            .unwrap();

        input_dim
    }

    fn forward(&mut self, z: ArrayViewD<f32>) -> ArrayD<f32> {
        let a = self.infer(z);

        let error = self.derivative(a.view());
        self.gradients
            .push(Axis(0), error.view())
            .expect("failed to push gradients to sigmoid layer");

        a
    }

    fn backward(&self, loss: ArrayViewD<f32>) -> ArrayD<f32> {
        &loss * &self.gradients
    }
}

impl Sigmoid {
    fn derivative(&mut self, a: ArrayViewD<f32>) -> ArrayD<f32> {
        let error = a.mapv(|el| el * (1. - el));

        error
    }
}
