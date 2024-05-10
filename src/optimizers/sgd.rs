use crate::neural_network::model::Model;
use crate::optimizers::optimizer::Optimizer;
use ndarray::IxDyn;

pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        SGD { learning_rate }
    }
}

impl Optimizer for SGD {
    fn prepare(&self, model: &mut Model, training_data_dim: IxDyn) {
        let mut next_input_dim = training_data_dim;

        for module in model.modules.iter_mut() {
            next_input_dim = module.prepare(next_input_dim);
        }
    }

    fn batch_size(&mut self) -> f32 {
        1.
    }
}

// impl SGD {
//     fn gradient_adjustment(
//         &self,
//         mut weights: ArrayViewMutD<f32>,
//         mut biases: ArrayViewMutD<f32>,
//         mut gradients: ArrayViewMutD<f32>,
//     ) {
//         gradients *= self.learning_rate;
//
//         weights *= weights.mul(&gradients);
//         biases *= biases.mul(&gradients);
//     }
// }
