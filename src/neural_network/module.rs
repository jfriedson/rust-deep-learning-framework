use ndarray::{ArrayD, ArrayViewD, IxDyn};

pub trait Module {
    fn infer(&self, input: ArrayViewD<f32>) -> ArrayD<f32>;

    fn prepare(&mut self, input_dim: IxDyn) -> IxDyn;

    fn forward(&mut self, input: ArrayViewD<f32>) -> ArrayD<f32>;

    fn backward(&self, loss: ArrayViewD<f32>) -> ArrayD<f32>;
}
