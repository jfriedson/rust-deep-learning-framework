use ndarray::{Array, ArrayView, IxDyn};
use crate::loss_functions::loss_function::LossFunction;

pub struct MSE {
}

impl MSE {
    pub fn new() -> Self {
        MSE {
        }
    }
}

impl LossFunction for MSE {
    fn forward(&self, predictions: ArrayView<f32, IxDyn>, truths: ArrayView<f32, IxDyn>) -> Array<f32, IxDyn> {
        let mut mse = &predictions - &truths;
        mse.mapv_inplace(|x| x.powi(2));

        mse
    }

    fn backward(&self, predictions: ArrayView<f32, IxDyn>, truths: ArrayView<f32, IxDyn>) -> Array<f32, IxDyn> {
        &predictions - &truths
    }
}
