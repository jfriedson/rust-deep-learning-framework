use crate::loss_functions::loss_function::LossFunction;
use crate::neural_network::model_builder::ModelBuilder;
use crate::neural_network::module::Module;
use crate::optimizers::optimizer::Optimizer;
use ndarray::{Array, ArrayView, ArrayView3, IxDyn};

pub struct Model {
    pub(crate) modules: Vec<Box<dyn Module>>,
    pub(crate) loss_fn: Box<dyn LossFunction>,
    pub(crate) optimizer: Box<dyn Optimizer>,
}

impl Model {
    pub fn builder() -> ModelBuilder {
        ModelBuilder::new()
    }

    pub fn forward(&self, input: ArrayView<f32, IxDyn>, training: bool) -> Array<f32, IxDyn> {
        let mut next_input = input.to_owned();

        for module in self.modules.iter() {
            next_input = module.forward(next_input.view(), training);
        }

        next_input
    }

    pub fn backward(&self, loss: ArrayView<f32, IxDyn>) {

    }

    pub fn train(&self, training_data: ArrayView3<f32> /* make dyn */, epochs: u32) {
        //self.optimizer.prepare(&self);
        for iteration in 0..epochs {
            let loss = self.optimizer.step(&self, training_data);

            println!("iteration: {} loss: {}", iteration + 1, loss);
        }
    }
}
