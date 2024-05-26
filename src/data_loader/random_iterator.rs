use ndarray::{ArrayD, ArrayViewD, Axis};

pub struct RandomIter<'a, A> {
    pub(crate) data_set_inputs: &'a ArrayD<A>,
    pub(crate) data_set_outputs: &'a ArrayD<A>,
    pub(crate) index_vec: Vec<usize>,
}

impl<'a, A> Iterator for RandomIter<'a, A> {
    type Item = (ArrayViewD<'a, A>, ArrayViewD<'a, A>);

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index_vec.pop()?;

        let sample_input = self.data_set_inputs.index_axis(Axis(0), index);
        let sample_output = self.data_set_outputs.index_axis(Axis(0), index);

        Some((sample_input, sample_output))
    }
}
