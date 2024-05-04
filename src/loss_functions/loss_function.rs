use ndarray::{ArrayD, ArrayViewD};

pub trait LossFunction {
    fn forward(
        &self,
        predictions: ArrayViewD<f32>,
        truths: ArrayViewD<f32>,
    ) -> ArrayD<f32>;

    fn backward(
        &self,
        predictions: ArrayViewD<f32>,
        truths: ArrayViewD<f32>,
    ) -> ArrayD<f32>;
}
