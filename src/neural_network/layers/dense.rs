use ndarray::{Array, Array1, Array2, ArrayView, Ix1, IxDyn};
use ndarray_rand::RandomExt;
use rand::distributions::Standard;
use crate::neural_network::module::Module;

pub struct Dense {
    weights: Array2<f32>,
    biases: Array1<f32>,
}

impl Dense {
    pub fn new(input_count: usize, output_count: usize) -> Self
    {
        assert!(input_count > 0, "number of inputs must be greater than 0");
        assert!(output_count > 0, "number of outputs must be greater than 0");

        let weights = Array2::<f32>::random((output_count, input_count), Standard);
        let biases = Array1::<f32>::zeros(output_count);

        Dense {
            weights,
            biases
        }
    }
}

impl Module for Dense {
    fn trainable(&self) -> bool {
        true
    }

    fn forward(&self, input: ArrayView<f32, IxDyn>) -> Array<f32, IxDyn> {
        let input_flattened = input.into_dimensionality::<Ix1>().unwrap();
        (&self.weights.dot(&input_flattened) + &self.biases).into_dyn()
    }

    // fn forward(&self, input: &ArrayView1<f32>) -> (Array1<f32>, Array1<f32>) {
    //     let z = &self.weights.dot(input) + &self.biases;
    // }
}