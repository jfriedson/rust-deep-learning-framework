use ndarray::iter::AxisIter;
use ndarray::{ArrayViewD, IxDyn};
use std::iter::Zip;

pub struct BatchIter<'a, A> {
    pub(crate) data_iter: Zip<AxisIter<'a, A, IxDyn>, AxisIter<'a, A, IxDyn>>,
    pub(crate) batch_size: usize,
}

impl<'a, A> Iterator for BatchIter<'a, A> {
    type Item = Vec<(ArrayViewD<'a, A>, ArrayViewD<'a, A>)>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut result: Self::Item = Vec::new();

        for _ in 0..self.batch_size {
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
