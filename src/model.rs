use crate::optimizers::optimizer::Optimizer;
use ndarray::{Array0, ArrayD, ArrayViewD};
use crate::neural_network::activations::leaky_relu::LeakyReLU;
use crate::neural_network::activations::sigmoid::Sigmoid;
use crate::neural_network::layers::dense::Dense;

pub struct Model {
    dense1: Dense,
    lr1: LeakyReLU,
    dense2: Dense,
    sigmoid: Sigmoid,
}

#[allow(unused)]
impl Model {
    pub fn new() -> Self {
        Model {
            dense1: Dense::new(2, 4),
            lr1: LeakyReLU::new(0.1),
            dense2: Dense::new(4, 4),
            sigmoid: Sigmoid::new(),
        }
    }

    pub fn infer(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        let mut x = self.dense1.infer(input);
        x = self.lr1.infer(x.view());
        x = self.dense2.infer(x.view());
        self.sigmoid.infer(x.view())
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

        let mut module_iter = self.modules.iter_mut().rev();
        if let Some(module) = module_iter.next() {
            next_loss = module.backward(loss);
        }

        for module in module_iter {
            next_loss = module.backward(next_loss.view());
        }
    }

    pub fn apply_gradients(&mut self, optimizer: &Box<dyn Optimizer>) {
        for module in self.modules.iter_mut() {
            module.apply_gradients(&optimizer);
        }
    }

    pub fn zero_gradients(&mut self) {
        for module in self.modules.iter_mut() {
            module.zero_gradients();
        }
    }
}
