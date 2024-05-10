use ndarray::iter::AxisIter;
use ndarray::{ArrayView, ArrayViewD, IxDyn};
use rand::seq::IteratorRandom;

pub struct BatchIter<'a, A> {
    pub(crate) data_iter: AxisIter<'a, A, IxDyn>,
    pub(crate) batch_size: usize,
}

impl<'a, A> Iterator for BatchIter<'a, A> {
    type Item = Vec<ArrayViewD<'a, A>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut result: Self::Item = Vec::new();

        for index in 0..self.batch_size {
            let data = self.data_iter.next();
            if data.is_none() {
                break;
            }

            result.push(data.unwrap());
        }

        if result.is_empty() {
            None?
        }

        Some(result)
    }
}

pub struct RandomIter<'a, A> {
    pub(crate) data_iter: AxisIter<'a, A, IxDyn>,
}

impl<'a, A> RandomIter<'a, A> {
    fn next(mut self) -> Option<ArrayView<'a, A, IxDyn>> {
        self.data_iter.choose(&mut rand::thread_rng())
    }
}

pub struct RandomBatchIter<'a, A> {
    pub(crate) data_iter: AxisIter<'a, A, IxDyn>,
    pub(crate) batch_size: usize,
}

impl<'a, A> RandomBatchIter<'a, A> {
    fn next(mut self) -> Option<Vec<ArrayView<'a, A, IxDyn>>> {
        let batch = self
            .data_iter
            .choose_multiple(&mut rand::thread_rng(), self.batch_size);

        if batch.len() == 0 {
            None?
        }

        Some(batch)
    }
}
