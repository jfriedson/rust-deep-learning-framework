use crate::loss_functions::loss_function::LossFunction;
use crate::neural_network::model::Model;
use crate::neural_network::module::Module;
use crate::optimizers::optimizer::Optimizer;
use std::mem::take;

pub struct ModelBuilder {
    modules: Vec<Box<dyn Module>>,
    loss_fn: Option<Box<dyn LossFunction>>,
    optimizer: Option<Box<dyn Optimizer>>,
}

impl ModelBuilder {
    pub fn new() -> Self {
        let modules = Vec::new();
        let loss_fn = None;
        let optimizer = None;

        ModelBuilder {
            modules,
            loss_fn,
            optimizer,
        }
    }

    pub fn add_module(&mut self, module: Box<dyn Module>) -> &mut Self {
        self.modules.push(module);

        self
    }

    pub fn set_loss_fn(&mut self, loss_fn: Box<dyn LossFunction>) -> &mut Self {
        self.loss_fn = Option::from(loss_fn);

        self
    }

    pub fn set_optimizer(&mut self, optimizer: Box<dyn Optimizer>) -> &mut Self {
        self.optimizer = Option::from(optimizer);

        self
    }

    pub fn build(&mut self) -> Model {
        Model {
            modules: take(&mut self.modules),
            loss_fn: take(&mut self.loss_fn).unwrap(),
            optimizer: take(&mut self.optimizer).unwrap(),
        }
    }
}
