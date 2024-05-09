use ndarray::{ArrayD, Axis, IxDyn};
use ndarray::iter::AxisIter;

pub struct DataLoader<A> {
    data_set: ArrayD<A>,
}

impl<A> DataLoader<A> {
    pub fn from_array(array: ArrayD<A>) -> Self {
        DataLoader {
            data_set: array,
        }
    }

    pub fn get_next_data_sample(&mut self) -> AxisIter<A, IxDyn> {
        self.data_set.axis_iter(Axis(0))
    }

    pub fn get_next_data_batch(&mut self) -> AxisIter<A, IxDyn> {
        self.data_set.axis_iter(Axis(0))
    }
}
