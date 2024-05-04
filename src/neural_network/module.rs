use ndarray::{ArrayD, ArrayViewD, IxDyn};

pub trait Module {
    fn infer(&self, input: ArrayViewD<f32>) -> ArrayD<f32>;

    fn prepare(&self, batch_size: usize, input_dim: IxDyn) -> IxDyn;

    fn forward(&self, input: ArrayViewD<f32>) -> ArrayD<f32>;

    fn backward(&self, loss: ArrayViewD<f32>) -> ArrayD<f32>;
}
