use ndarray::{ArrayViewD, IxDyn};
use ndarray::iter::AxisIter;

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
