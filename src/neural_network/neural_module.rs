pub trait NeuralModule {
    fn trainable(&self) -> bool;
}