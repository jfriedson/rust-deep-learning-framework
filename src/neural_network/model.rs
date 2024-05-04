use crate::loss_functions::loss_function::LossFunction;
use crate::neural_network::model_builder::ModelBuilder;
use crate::neural_network::module::Module;
use crate::optimizers::optimizer::Optimizer;
use ndarray::{ArrayD, ArrayViewD};

pub struct Model {
    pub(crate) modules: Vec<Box<dyn Module>>,
    pub(crate) loss_fn: Box<dyn LossFunction>,
    pub(crate) optimizer: Box<dyn Optimizer>,
}

impl Model {
    pub fn builder() -> ModelBuilder {
        ModelBuilder::new()
    }

    /*
     * The methods below are structured this way because copying memory is expensive.
     * Initializing next_input to 0s and manually stepping through the first iteration
     * should be quicker than copying the input array despite the increased function size.
     * Will test this hypothesis.
     */
    pub fn infer(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        let mut next_input = ArrayD::zeros(input.raw_dim());

        let mut module_iter = self.modules.iter();
        if let Some(module) = module_iter.next() {
            next_input = module.infer(input);
        }

        for module in module_iter {
            next_input = module.infer(next_input.view());
        }

        next_input
    }

    pub fn forward(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        let mut next_input = ArrayD::zeros(input.raw_dim());

        let mut module_iter = self.modules.iter();
        if let Some(module) = module_iter.next() {
            next_input = module.forward(input);
        }

        for module in module_iter {
            next_input = module.forward(next_input.view());
        }

        next_input
    }

    pub fn backward(&self, loss: ArrayViewD<f32>) {
        let mut next_loss = ArrayD::zeros(loss.raw_dim());

        let mut module_iter = self.modules.iter();
        if let Some(module) = module_iter.next() {
            next_loss = module.forward(loss);
        }

        for module in module_iter {
            next_loss = module.forward(next_loss.view());
        }

        //println!("{:?}", next_loss);
    }

    pub fn train(&self, training_data: ArrayViewD<f32>, epochs: usize) {
        self.optimizer.prepare(&self, training_data.raw_dim());

        for iteration in 0..epochs {
            let loss = self.optimizer.step(&self, &training_data);

            println!("iteration: {} loss: {}", iteration + 1, loss);
        }
    }
}
