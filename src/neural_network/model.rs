use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use crate::loss_functions::loss_function::LossFunction;
use crate::neural_network::model_builder::ModelBuilder;
use crate::neural_network::module::Module;
use crate::optimizers::optimizer::Optimizer;
use ndarray::{Array0, ArrayD, ArrayViewD};

pub struct Model {
    pub(crate) modules: Vec<Box<dyn Module>>,
    pub(crate) loss_fn: Box<dyn LossFunction>,
    pub(crate) optimizer: Box<dyn Optimizer>,
}

impl Model {
    pub fn builder() -> ModelBuilder {
        ModelBuilder::new()
    }

    pub fn infer(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        let mut next_input = Array0::<f32>::into_dyn(Default::default());

        let mut module_iter = self.modules.iter();
        if let Some(module) = module_iter.next() {
            next_input = module.infer(input);
        }

        for module in module_iter {
            next_input = module.infer(next_input.view());
        }

        next_input
    }

    pub fn forward(&mut self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        let mut next_input = Array0::<f32>::into_dyn(Default::default());

        let mut module_iter = self.modules.iter_mut();
        if let Some(module) = module_iter.next() {
            next_input = module.forward(input);
        }

        for module in module_iter {
            next_input = module.forward(next_input.view());
        }

        next_input
    }

    pub fn backward(&mut self, loss: ArrayViewD<f32>) {
        let mut next_loss = Array0::<f32>::into_dyn(Default::default());

        let mut module_iter = self.modules.iter_mut();
        if let Some(module) = module_iter.next() {
            next_loss = module.forward(loss);
        }

        for module in module_iter {
            next_loss = module.forward(next_loss.view());
        }
    }
}
