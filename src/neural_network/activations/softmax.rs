use crate::neural_network::neural_component::NeuralComponent;
use crate::optimizers::optimizer::Optimizer;
use ndarray::{concatenate, Array1, ArrayD, ArrayViewD, Axis, Dimension, Ix1};

pub struct Softmax {
    gradients: ArrayD<f32>,
}

#[allow(unused)]
impl Softmax {
    pub fn new() -> Self {
        let gradients = Array1::<f32>::zeros(0).into_dyn();

        Softmax { gradients }
    }

    fn derivative(&mut self, a: ArrayViewD<f32>) -> ArrayD<f32> {
        let size = a.raw_dim().size();

        let mut tiled = a.to_owned().insert_axis(Axis(0));
        for i in 1..size {
            tiled = concatenate(Axis(0), &[tiled.view(), a.view().insert_axis(Axis(0))]).unwrap();
        }

        &tiled * (ndarray::Array2::<f32>::eye(size) - tiled.t())
    }
}

impl NeuralComponent for Softmax {
    fn infer(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        let max = input
            .iter()
            .reduce(|max: &f32, x: &f32| if x > max { x } else { max })
            .unwrap();
        let exp = input.mapv(|x| (x - max).exp());

        &exp / (exp.sum())
    }

    fn forward(&mut self, z: ArrayViewD<f32>) -> ArrayD<f32> {
        let a = self.infer(z);

        self.gradients = self.derivative(a.view());

        a
    }

    fn backward(&mut self, losses: ArrayViewD<f32>) -> ArrayD<f32> {
        let gradients_flat = self
            .gradients
            .view()
            .into_shape((losses.raw_dim().size(), losses.raw_dim().size()))
            .unwrap();

        let result = losses
            .into_dimensionality::<Ix1>()
            .unwrap()
            .dot(&gradients_flat);

        result.into_dyn()
    }

    fn apply_gradients(&mut self, _optimizer: &dyn Optimizer) {
        // not trainable
    }

    fn zero_gradients(&mut self) {
        let gradient_shape = self.gradients.raw_dim();
        self.gradients = ArrayD::<f32>::zeros(gradient_shape);
    }
}
