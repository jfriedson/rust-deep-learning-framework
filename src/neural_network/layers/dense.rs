use crate::neural_network::neural_component::NeuralComponent;
use crate::optimizers::optimizer::Optimizer;
use ndarray::{Array1, Array2, ArrayD, ArrayViewD, Axis, Ix1};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

pub struct Dense {
    weights: Array2<f32>,
    biases: Array1<f32>,

    inputs: Array2<f32>,

    gradients: Array1<f32>,
    gradient_velocities: Array1<f32>,
}

#[allow(unused)]
impl Dense {
    pub fn new(input_count: usize, output_count: usize) -> Self {
        assert!(input_count > 0, "number of inputs must be greater than 0");
        assert!(output_count > 0, "number of outputs must be greater than 0");

        let kaiming_variance = (2. / input_count as f32).sqrt();
        let weights = Array2::<f32>::random(
            (output_count, input_count),
            Normal::new(0., kaiming_variance).unwrap(),
        );
        let biases = Array1::<f32>::from_elem(output_count, 0.001);

        let inputs = Array2::<f32>::zeros((0, input_count));

        let gradients = Array1::<f32>::zeros(output_count);
        let gradient_velocities = Array1::<f32>::zeros(output_count);

        Dense {
            weights,
            biases,
            inputs,
            gradients,
            gradient_velocities,
        }
    }
}

impl NeuralComponent for Dense {
    fn infer(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        let input_flattened = input.into_dimensionality::<Ix1>().unwrap();

        let z = &self.weights.dot(&input_flattened) + &self.biases;

        z.into_dyn()
    }

    fn forward(&mut self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        let input_flattened = input.into_dimensionality::<Ix1>().unwrap();

        self.inputs = input_flattened.insert_axis(Axis(0)).to_owned();

        self.infer(input_flattened.into_dyn())
    }

    fn backward(&mut self, losses: ArrayViewD<f32>) -> ArrayD<f32> {
        let losses_dim = losses.into_dimensionality::<Ix1>().unwrap();

        self.gradients = losses_dim.to_owned();

        losses_dim.dot(&self.weights).into_dyn()
    }

    fn apply_gradients(&mut self, optimizer: &dyn Optimizer) {
        optimizer.adjust_gradients(self.gradients.view_mut().into_dyn(), self.gradient_velocities.view_mut().into_dyn());
        self.biases -= &self.gradients;

        let grads_dim2 = self.gradients.view().insert_axis(Axis(0));
        let mut weight_deltas = grads_dim2.t().dot(&self.inputs);
        optimizer.adjust_weight_deltas(
            weight_deltas.view_mut().into_dyn(),
            self.weights.view().into_dyn(),
        );
        self.weights -= &weight_deltas;
    }

    fn zero_gradients(&mut self) {
        self.gradients = Array1::<f32>::zeros(self.gradients.raw_dim());
        self.inputs = Array2::<f32>::zeros(self.inputs.raw_dim());
    }
}
