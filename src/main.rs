mod data;
mod model; 
mod training;
mod inference; 

use crate::{model::ModelConfig, training::TrainingConfig}; 
use burn::{
    data::dataset::Dataset, optim::AdamConfig, prelude::*
};

fn main() {
    let device = Device::flex(); 
    // Creation of two tensors, the first with explicit values and the second one with ones, with the same shape as the first 
    // let tensor_1 = Tensor::<2>::from_data([[2.,3.], [4.,5.]], &device); 
    // let tensor_2 = Tensor::<2>::ones_like(&tensor_1); 

    // println!("Result: {}", tensor_1 + tensor_2);
    // println!("Hello, world!");  

    // let model = ModelConfig::new(10, 512).init(&device); 
    // println!("{model}");

    let artifact_dir = "target/guide"; 
    crate::training::train(artifact_dir,TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()), device.clone());
    crate::inference::infer(artifact_dir, device, burn::data::dataset::vision::MnistDataset::test().get(100).unwrap());
}

