use ndarray::iter::AxisIter;
use ndarray::{ArrayD, Axis, IxDyn};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::iter::Zip;

use crate::data_loader::batch_iterator::BatchIter;
use crate::data_loader::random_iterator::RandomIter;

pub struct DataLoader<A> {
    data_set_inputs: ArrayD<A>,
    data_set_outputs: ArrayD<A>,
}

#[allow(unused)]
impl<A> DataLoader<A> {
    pub fn from_arrays(data_set_inputs: ArrayD<A>, data_set_outputs: ArrayD<A>) -> Self {
        assert_eq!(data_set_inputs.len_of(Axis(0)), data_set_outputs.len_of(Axis(0)), "input and output sample count in data set must be equal");

        DataLoader {
            data_set_inputs,
            data_set_outputs,
        }
    }

    pub fn get_input_dim(&self) -> IxDyn {
        self.data_set_inputs.raw_dim()
    }

    pub fn iter(&mut self) -> Zip<AxisIter<'_, A, IxDyn>, AxisIter<'_, A, IxDyn>> {
        self.data_set_inputs
            .axis_iter(Axis(0))
            .zip(self.data_set_outputs.axis_iter(Axis(0)))
    }

    pub fn batch_iter(&mut self, batch_size: usize) -> BatchIter<A> {
        BatchIter {
            data_iter: self.iter(),
            batch_size,
        }
    }

    pub fn rand_iter(&mut self) -> RandomIter<A> {
        let vec_size = self.data_set_inputs.len_of(Axis(0));
        let mut index_vec: Vec<usize> = (0..vec_size).collect();
        index_vec.shuffle(&mut thread_rng());

        RandomIter {
            data_set_inputs: &self.data_set_inputs,
            data_set_outputs: &self.data_set_outputs,
            index_vec,
        }
    }
}
