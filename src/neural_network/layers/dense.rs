use crate::neural_network::module::Module;
use ndarray::{Array1, Array2, ArrayD, ArrayViewD, Dimension, Ix1, IxDyn};
use ndarray_rand::RandomExt;
use rand::distributions::Standard;

pub struct Dense {
    weights: Array2<f32>,
    biases: Array1<f32>,

    input_aggr: Array2<f32>,
}

impl Dense {
    pub fn new(input_count: usize, output_count: usize) -> Self {
        assert!(input_count > 0, "number of inputs must be greater than 0");
        assert!(output_count > 0, "number of outputs must be greater than 0");

        let weights = Array2::<f32>::random((output_count, input_count), Standard);
        let biases = Array1::<f32>::zeros(output_count);

        let input_aggr = Array2::<f32>::zeros((0,0));

        Dense { weights, biases, input_aggr }
    }
}

impl Module for Dense {
    fn infer(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        if input.ndim() != 1 {
            panic!("for now, fully connected layers only support 1 dimensional data")
        }

        let input_flattened = input.into_dimensionality::<Ix1>().unwrap();

        let z = &self.weights.dot(&input_flattened) + &self.biases;

        z.into_dyn()
    }

    fn prepare(&mut self, batch_size: usize, input_dim: IxDyn) -> IxDyn {
        self.input_aggr = Array2::zeros((batch_size, input_dim.size()));

        // replace this with something better, like a struct member
        self.biases.raw_dim().into_dyn()
    }

    fn forward(&self, input: ArrayViewD<f32>) -> ArrayD<f32> {
        // TODO: append input

        self.infer(input)
    }

    fn backward(&self, loss: ArrayViewD<f32>) -> ArrayD<f32> {
        // TODO: calculate delta

        Array1::<f32>::from_elem(self.biases.raw_dim(), -0.01).into_dyn()
    }
}
