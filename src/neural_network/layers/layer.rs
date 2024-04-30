use ndarray::{Array1, ArrayBase, Data, Ix1};

pub trait Layer<S>
where
    S: Data<Elem = f32>,
{
    fn infer(&self, input: ArrayBase<S, Ix1>) -> Array1<f32>
    where
        S: Data<Elem = f32>;
}
