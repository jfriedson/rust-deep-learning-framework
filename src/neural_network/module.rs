use ndarray::{ArrayD, ArrayViewD, IxDyn};
use crate::optimizers::optimizer::Optimizer;

pub trait Module {
    fn infer(&self, input: ArrayViewD<f32>) -> ArrayD<f32>;

    fn prepare(&mut self, input_dim: IxDyn) -> IxDyn;

    fn forward(&mut self, input: ArrayViewD<f32>) -> ArrayD<f32>;

    fn backward(&mut self, loss: ArrayViewD<f32>) -> ArrayD<f32>;

    fn apply_gradients(&mut self, optimizer: &Box<dyn Optimizer>);

    fn zero_gradients(&mut self);
}
