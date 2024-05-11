use crate::neural_network::module::Module;
use ndarray::{Array1, Array2, ArrayD, ArrayViewD, Axis, Dimension, Ix1, IxDyn};
use ndarray_rand::RandomExt;
use rand::distributions::Standard;

pub struct Dense {
    weights: Array2<f32>,
    biases: Array1<f32>,

    inputs: Array2<f32>,
    deltas: Array1<f32>,
}

impl Dense {
    pub fn new(input_count: usize, output_count: usize) -> Self {
        assert!(input_count > 0, "number of inputs must be greater than 0");
        assert!(output_count > 0, "number of outputs must be greater than 0");

        let weights = Array2::<f32>::random((output_count, input_count), Standard);
        let biases = Array1::<f32>::zeros(output_count);

        let inputs = Array2::<f32>::zeros((0, input_count));
        let deltas = Array1::<f32>::zeros(0);

        Dense {
            weights,
            biases,
            inputs,
            deltas,
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
        self.inputs = Array2::zeros((0, input_dim.size()));

        self.biases.raw_dim().into_dyn()
    }

    fn forward(&mut self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        let input_flattened = input.into_dimensionality::<Ix1>().unwrap();

        self.inputs.push(Axis(0), input_flattened).unwrap();

        self.infer(input_flattened.into_dyn())
    }

    fn backward(&self, loss: ArrayViewD<f32>) -> ArrayD<f32> {
        // TODO: calculate delta

        Array1::<f32>::from_elem(self.biases.raw_dim(), -0.01).into_dyn()
    }
}
