use crate::neural_network::model::Model;
use crate::neural_network::module::Module;
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
