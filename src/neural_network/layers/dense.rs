use ndarray::{Array1, Array2, ArrayView1};
use ndarray_rand::RandomExt;
use rand::distributions::Standard;
use crate::neural_network::layers::layer::Layer;

pub struct Dense {
    weights: Array2<f32>,
    biases: Array1<f32>,
}

impl Layer for Dense {
    fn new(input_count: usize, output_count: usize) -> Self {
        assert!(input_count > 0, "number of inputs must be greater than 0");
        assert!(output_count > 0, "number of outputs must be greater than 0");

        let weights = Array2::<f32>::random((output_count, input_count), Standard);
        let biases = Array1::<f32>::zeros(output_count);

        Dense {
            weights,
            biases
        }
    }

    fn infer(&self, input: &ArrayView1<f32>) -> Array1<f32> {
        &self.weights.dot(input) + &self.biases
    }

    // fn forward(&self, input: &ArrayView1<f32>) -> (Array1<f32>, Array1<f32>) {
    //     let z = &self.weights.dot(input) + &self.biases;
    // }
}