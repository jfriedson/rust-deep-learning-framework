use crate::optimizers::optimizer::Optimizer;
use ndarray::{Array1, ArrayD, ArrayViewD, Axis};

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
        let diff_max = &input - *max;
        let exp_diff_max = diff_max.mapv(|x| x.exp());

        &exp_diff_max / (&exp_diff_max.sum_axis(Axis(0)))
    }

    pub fn forward(&mut self, z: ArrayViewD<f32>) -> ArrayD<f32> {
        let a = self.infer(z);

        self.gradients = self.derivative(a.view());

        a
    }

    pub fn backward(&mut self, losses: ArrayViewD<f32>) -> ArrayD<f32> {
        &losses * &self.gradients
    }

    pub fn apply_gradients(&mut self, _optimizer: &Box<dyn Optimizer>) {
        // not trainable, do nothing
    }

    pub fn zero_gradients(&mut self) {
        let gradient_shape = self.gradients.raw_dim();
        self.gradients = ArrayD::<f32>::zeros(gradient_shape);
    }

    pub fn derivative(&mut self, a: ArrayViewD<f32>) -> ArrayD<f32> {
        &a * (1. - &a)
    }
}
