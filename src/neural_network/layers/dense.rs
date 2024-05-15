use std::ops::{Sub};
use crate::neural_network::module::Module;
use ndarray::{Array1, Array2, ArrayD, ArrayViewD, Axis, Dimension, Ix1, IxDyn};
use ndarray_rand::RandomExt;
use rand::distributions::Standard;
use crate::optimizers::optimizer::Optimizer;

pub struct Dense {
    weights: Array2<f32>,
    biases: Array1<f32>,

    inputs: Array2<f32>,
    gradients: Array2<f32>,
}

impl Dense {
    pub fn new(input_count: usize, output_count: usize) -> Self {
        assert!(input_count > 0, "number of inputs must be greater than 0");
        assert!(output_count > 0, "number of outputs must be greater than 0");

        let weights = Array2::<f32>::random((output_count, input_count), Standard);
        let biases = Array1::<f32>::zeros(output_count);

        let inputs = Array2::<f32>::zeros((0, input_count));
        let gradients = Array2::<f32>::zeros((output_count, input_count));

        Dense {
            weights,
            biases,
            inputs,
            gradients,
        }
    }
}

impl Module for Dense {
    fn infer(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        let input_flattened = input.into_dimensionality::<Ix1>().unwrap();

        let z = &self.weights.dot(&input_flattened) + &self.biases;

        z.into_dyn()
    }

    fn prepare(&mut self, input_dim: IxDyn) -> IxDyn {
        debug_assert_eq!(
            self.inputs.dim(),
            Array2::<f32>::zeros((0, input_dim.size())).dim(),
            "input array shape does not match in preparation of dense layer"
        );

        self.biases.raw_dim().into_dyn()
    }

    fn forward(&mut self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        let input_flattened = input.into_dimensionality::<Ix1>().unwrap();

        self.inputs.push(Axis(0), input_flattened).unwrap();
        self.inputs = self.inputs.sum_axis(Axis(0)).insert_axis(Axis(0));

        self.infer(input_flattened.into_dyn())
    }

    fn backward(&mut self, losses: ArrayViewD<f32>) -> ArrayD<f32> {
        let losses_dim = losses.into_dimensionality::<Ix1>().unwrap().insert_axis(Axis(0));

        let prev_losses = &losses_dim.dot(&self.weights);

        self.gradients = losses_dim.to_owned();

        prev_losses.clone().into_dyn()
    }

    fn apply_gradients(&mut self, optimizer: &Box<dyn Optimizer>) {
        optimizer.adjust_gradients(self.gradients.view_mut().into_dyn());

        let adjustment = &self.inputs.t().dot(&self.gradients);

        self.weights = self.weights.clone().sub(adjustment.t());
        self.biases = self.biases.clone().sub(self.gradients.clone().remove_axis(Axis(0)));

        self.gradients = Array2::<f32>::zeros(self.gradients.raw_dim());
        self.inputs = Array2::<f32>::zeros(self.inputs.raw_dim());
    }
}
