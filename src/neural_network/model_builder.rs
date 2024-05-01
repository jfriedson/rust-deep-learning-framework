use crate::loss_functions::loss_function::LossFunction;
use crate::neural_network::model::Model;
use crate::neural_network::module::Module;

pub struct ModelBuilder {
    modules: Vec<Box<dyn Module>>,
    loss_fn: Option<Box<dyn LossFunction>>,
    // TODO: add optimizer
}

impl ModelBuilder {
    pub fn new() -> Self {
        let modules = Vec::new();
        let loss_fn = Option::None;

        ModelBuilder {
            modules,
            loss_fn
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

    pub fn build(self) -> Model {
        //let modules = &self.modules;
        //let loss_fn = &self.loss_fn;

        Model {
            modules: self.modules,
            loss_fn: self.loss_fn.unwrap(),

            training: false,
        }
    }
}
