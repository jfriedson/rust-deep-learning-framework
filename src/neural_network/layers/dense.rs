use ndarray::{Array1, Array2, ArrayBase, Data, Ix1};
use ndarray_rand::RandomExt;
use rand::distributions::Standard;
use crate::neural_network::layers::layer::Layer;
use crate::neural_network::neural_module::NeuralModule;

pub struct Dense {
    weights: Array2<f32>,
    biases: Array1<f32>,
}

impl NeuralModule for Dense {
    fn trainable(&self) -> bool {
        true
    }
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

impl<S> Layer<S> for Dense
where
    S: Data<Elem = f32>,
{
    fn infer(&self, input: ArrayBase<S, Ix1>) -> Array1<f32>
    where
        S: Data<Elem = f32>,
    {
        &self.weights.dot(&input) + &self.biases
    }

    // fn forward(&self, input: &ArrayView1<f32>) -> (Array1<f32>, Array1<f32>) {
    //     let z = &self.weights.dot(input) + &self.biases;
    // }
}