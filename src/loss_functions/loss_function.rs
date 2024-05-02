use ndarray::{Array, ArrayView, IxDyn};

pub trait LossFunction {
    fn forward(
        &self,
        predictions: ArrayView<f32, IxDyn>,
        truths: ArrayView<f32, IxDyn>,
    ) -> Array<f32, IxDyn>;
    fn backward(
        &self,
        predictions: ArrayView<f32, IxDyn>,
        truths: ArrayView<f32, IxDyn>,
    ) -> Array<f32, IxDyn>;
}
