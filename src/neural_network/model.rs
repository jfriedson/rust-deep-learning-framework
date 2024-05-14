use crate::neural_network::module::Module;
use ndarray::{Array0, ArrayD, ArrayViewD, ArrayViewMutD};

pub struct Model {
    pub(crate) modules: Vec<Box<dyn Module>>,
}

impl Model {
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

        let mut module_iter = self.modules.iter_mut().rev();
        if let Some(module) = module_iter.next() {
            next_loss = module.backward(loss);
        }

        for module in module_iter {
            next_loss = module.backward(next_loss.view());
        }
    }

    pub fn apply_gradients(&mut self, gradient_adjuster: fn(ArrayViewMutD<f32>)) {
        for module in self.modules.iter_mut() {
            module.apply_gradients(gradient_adjuster);
        }
    }
}
