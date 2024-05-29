use crate::optimizers::optimizer::Optimizer;
use ndarray::{Array1, ArrayD, ArrayViewD, Axis, concatenate, Dimension, Ix1};

pub struct Softmax {
    gradients: ArrayD<f32>,
}

#[allow(unused)]
impl Softmax {
    pub fn new() -> Self {
        let gradients = Array1::<f32>::zeros(0).into_dyn();

        Softmax { gradients }
    }

    pub fn infer(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        let max = input
            .iter()
            .reduce(|max: &f32, x: &f32| if (x > max) { x } else { max })
            .unwrap();
        let exp = input.mapv(|x| (x - max).exp());

        &exp / (exp.sum())
    }

    pub fn forward(&mut self, z: ArrayViewD<f32>) -> ArrayD<f32> {
        let a = self.infer(z);

        self.gradients = self.derivative(a.view());

        a
    }

    pub fn backward(&mut self, losses: ArrayViewD<f32>) -> ArrayD<f32> {
        let gradients_flat = self.gradients.view().into_shape((losses.raw_dim().size(), losses.raw_dim().size())).unwrap();

        let result = losses.into_dimensionality::<Ix1>().unwrap().dot(&gradients_flat);

        result.into_dyn()
    }

    pub fn apply_gradients(&mut self, _optimizer: &Box<dyn Optimizer>) {
        // not trainable, do nothing
    }

    pub fn zero_gradients(&mut self) {
        let gradient_shape = self.gradients.raw_dim();
        self.gradients = ArrayD::<f32>::zeros(gradient_shape);
    }

    pub fn derivative(&mut self, a: ArrayViewD<f32>) -> ArrayD<f32> {
        let size = a.raw_dim().size();

        let mut tiled = a.to_owned().insert_axis(Axis(0));
        for i in 1..size {
            tiled = concatenate(Axis(0), &[tiled.view(), a.view().insert_axis(Axis(0))]).unwrap();
        }

        &tiled * (ndarray::Array2::<f32>::eye(size) - tiled.t())
    }
}
