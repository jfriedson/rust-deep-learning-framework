use ndarray::{ArrayD, ArrayViewD, Axis};

pub struct RandomIter<'a, A> {
    pub(crate) data_set: &'a ArrayD<A>,
    pub(crate) index_vec: Vec<usize>,
}

impl<'a, A> Iterator for RandomIter<'a, A> {
    type Item = ArrayViewD<'a, A>;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index_vec.pop()?;

        Some(self.data_set.index_axis(Axis(0), index))
    }
}
