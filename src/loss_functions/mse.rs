use ndarray::{Array, ArrayBase, Data, Dimension};
use crate::loss_functions::loss_function::LossFunction;

pub struct MSE {
}

impl MSE {
    pub fn new() -> Self {
        MSE {
        }
    }
}

impl<S, D> LossFunction<S, D> for MSE
where
    S: Data<Elem = f32>,
    D: Dimension,
{
    fn forward(&self, predictions: ArrayBase<S, D>, truths: ArrayBase<S, D>) -> Array<f32, D> {
        let mut mse = &predictions - &truths;
        mse.mapv_inplace(|x| x.powi(2));

        mse
    }

    fn backward(&self, predictions: ArrayBase<S, D>, truths: ArrayBase<S, D>) -> Array<f32, D> {
        &predictions - &truths
    }
}
