# Inference 
2 things are needed to load weights for a model: the model's record and the model's config. Since parameter in Burn are lazy initialized, no allocation and GPU/CPU kernels are executed by the `ModelConfig::init` function. The weights are initilized when used for the first time, therefore can safely use `config.init(device).load_record(record)` without any meaningful performance cost. 

Simple `infer` method to load trained model: 
```rust
use crate::{data::MnistBatcher, training::TrainingConfig};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
    store::ModuleRecord,
};

pub fn infer(artifact_dir: &str, device: impl Into<Device>, item: MnistItem) {
    let device = device.into();
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = ModuleRecord::load(format!("{artifact_dir}/model"))
        .expect("Trained model should exist; run train first");

    let model = config.model.init(&device).load_record(record);

    let label = item.label;
    let batcher = MnistBatcher::default();
    let batch = batcher.batch(vec![item], &device);
    let output = model.forward(batch.images);
    let predicted: u8 = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {predicted} Expected {label}");
}
``` 

The first step is to load the configuraion of the training to fetch the correct model configuration. Then, we can load the saved record from its burnpack file. Finally, we can init the model with the configuration and apply the record. For simplicity we can use the same batcher used during the training to pass from a Mniistltm to a tensor. 

By running the infer function, see the prediction of model. 

Add the call to `infer` to the `main.rs` file aftter the `train` function call: 
```rust
    crate::inference::infer(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
``` 

The number `42` is the index of the image in the MNIST dataset. 

