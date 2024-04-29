use ndarray::{Array, ArrayBase, Data, Dimension};

pub trait LossFunction<S, D>
where
    S: Data<Elem = f32>,
    D: Dimension,
{
    fn forward(&self, predictions: ArrayBase<S, D>, truths: ArrayBase<S, D>) -> Array<f32, D>;
    fn backward(&self, predictions: ArrayBase<S, D>, truths: ArrayBase<S, D>) -> Array<f32, D>;
}
