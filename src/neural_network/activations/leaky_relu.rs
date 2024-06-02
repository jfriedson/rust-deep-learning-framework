use ndarray::{Array1, ArrayD, ArrayViewD};

pub struct LeakyReLU {
    negative_slope: f32,

    gradients: ArrayD<f32>,
}

#[allow(unused)]
impl LeakyReLU {
    pub fn new(negative_slope: f32) -> Self {
        let gradients = Array1::<f32>::zeros(0).into_dyn();

        LeakyReLU {
            negative_slope,

            gradients,
        }
    }

    pub fn infer(&self, z: ArrayViewD<f32>) -> ArrayD<f32> {
        z.mapv(|el| f32::max(el, el * self.negative_slope))
    }

    pub fn forward(&mut self, z: ArrayViewD<f32>) -> ArrayD<f32> {
        let a = self.infer(z.view());

        self.gradients = self.derivative(z);

        a
    }

    pub fn backward(&mut self, losses: ArrayViewD<f32>) -> ArrayD<f32> {
        &losses * &self.gradients
    }

    pub fn zero_gradients(&mut self) {
        let gradient_shape = self.gradients.raw_dim();
        self.gradients = ArrayD::<f32>::zeros(gradient_shape);
    }

    pub fn derivative(&mut self, a: ArrayViewD<f32>) -> ArrayD<f32> {
        a.mapv(|el| if el > 0. { 1. } else { self.negative_slope })
    }
}
