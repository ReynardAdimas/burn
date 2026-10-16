# Data 
Burn provides a library useful dataset sources and transformations, such as Hugging Face dataset utilities that allow to download and store data into an SQLite database for extremely efficient data streaming and storage. 

To iterate over a dataset efficiently, we will define a struct which will implement the `Batcher` trait. The goal of a batcher is to map individual dataset items into a batched tensor that can be used as input to our previously defined model. 

src/data.rs 
```rust
use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem}, 
    prelude::*
};

#[derive(Clone, Debug)]
pub struct MnistBatcher{} 
``` 

This bathcer is pretty straightforward, as it only defines a struct that will implement the `Batcher` trait. Its batching method receives Burn's runtime `Device`, so the tensors it creates are placed on the device selected by the caller. 

Implementing the batching logic: 
```rust
#[derive(Clone, Debug)]
pub struct MnistBatch {
    pub images: Tensor<3>, 
    pub targets: Tensor<1, Int>
}

impl Batcher<MnistItem, MnistBatch> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &Device) -> MnistBatch {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image))
            .map(|data| Tensor::<2>::from_data(data, device))
            .map(|tensor| tensor.reshape([1,28,28]))
            .map(|tensor| ((tensor / 255) - 0.1307)/0.3081)
            .collect(); 

        let targets = items
            .iter()
            .map(|item| Tensor::<1, Int>::from_data([item.label as i64], device))
            .collect(); 

        let images = Tensor::cat(images, 0); 
        let targets = Tensor::cat(targets, 0); 

        MnistBatch { images, targets }
    }
}
```

