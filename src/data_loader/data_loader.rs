use crate::data_loader::data_loader_iter::{BatchIter, RandomBatchIter};
use ndarray::iter::AxisIter;
use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn};
use rand::seq::IteratorRandom;

pub struct DataLoader<A> {
    data_set: ArrayD<A>,
}

impl<A> DataLoader<A> {
    pub fn from_array(array: ArrayD<A>) -> Self {
        DataLoader { data_set: array }
    }

    pub fn get_data_dim(&self) -> IxDyn {
        self.data_set.raw_dim()
    }

    pub fn shuffle_data(&mut self, batch_size: usize) -> Option<ArrayViewD<A>> {
        self.data_set
            .axis_iter(Axis(0))
            .choose(&mut rand::thread_rng())
    }

    pub fn batch_iter(&mut self, batch_size: usize) -> BatchIter<A> {
        BatchIter {
            data_iter: self.data_set.axis_iter(Axis(0)),
            batch_size,
        }
    }

    pub fn iter(&mut self) -> AxisIter<A, IxDyn> {
        self.data_set.axis_iter(Axis(0))
    }
}
