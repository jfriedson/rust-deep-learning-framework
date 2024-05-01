use ndarray::{Array, ArrayView, IxDyn};

pub trait Module {
    fn trainable(&self) -> bool;

    fn forward(&self, input: ArrayView<f32, IxDyn>) -> Array<f32, IxDyn>;
}
