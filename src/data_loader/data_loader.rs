use ndarray::iter::AxisIter;
use ndarray::{ArrayD, Axis, IxDyn, RemoveAxis};
use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::data_loader::batch_iterator::BatchIter;
use crate::data_loader::random_iterator::RandomIter;

pub struct DataLoader<A> {
    data_set: ArrayD<A>,
}

impl<A> DataLoader<A> {
    pub fn from_array(array: ArrayD<A>) -> Self {
        DataLoader { data_set: array }
    }

    pub fn get_data_dim(&self) -> IxDyn {
        self.data_set.raw_dim().remove_axis(Axis(0))
    }

    pub fn iter(&mut self) -> AxisIter<A, IxDyn> {
        self.data_set.axis_iter(Axis(0))
    }

    pub fn batch_iter(&mut self, batch_size: usize) -> BatchIter<A> {
        BatchIter {
            data_iter: self.data_set.axis_iter(Axis(0)),
            batch_size,
        }
    }

    pub fn rand_iter(&mut self) -> RandomIter<A> {
        let vec_size = self.data_set.axis_iter(Axis(0)).len();
        let mut index_vec: Vec<usize> = (0..vec_size).collect();
        index_vec.shuffle(&mut thread_rng());

        RandomIter {
            data_set: &self.data_set,
            index_vec,
        }
    }
}
