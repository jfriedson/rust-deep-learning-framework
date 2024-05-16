use crate::loss_functions::loss_function::LossFunction;
use crate::neural_network::model::Model;
use crate::neural_network::module::Module;
use crate::optimizers::optimizer::Optimizer;
use std::mem::take;

pub struct ModelBuilder {
    modules: Vec<Box<dyn Module>>,
}

impl ModelBuilder {
    pub fn new() -> Self {
        let modules = Vec::new();

        ModelBuilder { modules }
    }

    pub fn add_module(&mut self, module: Box<dyn Module>) -> &mut Self {
        self.modules.push(module);

        self
    }

    pub fn build(&mut self) -> Model {
        Model {
            modules: take(&mut self.modules),
        }
    }
}
