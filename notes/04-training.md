# Training 

Instead of a simple tensor, the model should output an item that can be understood by the learner,  a struct whose responsibility is to apply an optimizer to the model. The output struct is used for all metrics calculated during the training. Therefore it should include all the necessary information to calculate any metric that you want for a task. 

Burn provides 2 basic output types: `ClassificationOutput` and `RegressionOutput`. They implement the necessary trait to be used with metrics. It is possible to create your own item, but it is beyond the scope of this guide. 

Since the MNIST task is a classification problem, we will use the `ClassificationOutput` type : 
```rust
use std::path::PathBuf; 

use crate::{
    data::{MnistBatch, MnistBatcher}, 
    model::{Model, ModelConfig}
}; 

use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset}, 
    nn::loss::CrossEntropyLossConfig, 
    optim::AdamConfig, 
    prelude::*, 
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
        metric::{AccuracyMetric, LossMetric}
    }
};

// ANCHOR : classification 
impl Model {
    pub fn forward_classification(&self, images: Tensor<3>, targets:Tensor<1,Int>) -> ClassificationOutput {
        let output = self.forward(images); 
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    } 
}
``` 

Employ the cross-entropy loss module for loss calculation, without th inclusion of any padding token. Then, return the classification output containing the loss, the output tensor with all logits and the targets. 

Note : Tensor operations receive owned tensors as input. For reusing a tensor multiple items, use the `clone()` function. But this process won't involve actual copying of the tensor data. Instead, it will simply indicate that the tensor is employed in multiple instances, implying that certain operations won't be performed in place. The API has been designed with owned tensors to optimize performance. 

Implementation of both the training and validation steps : 
```rust 
impl TrainStep for Model {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: MnistBatch) -> TrainOutput<ClassificationOutput> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl InferenceStep for Model {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: MnistBatch) -> ClassificationOutput {
        self.forward_classification(batch.images, batch.targets)
    }
}

``` 

The `Input` and `Output` are a associated types of `TrainStep` and `InferenceStep` as `MnistBatch` and `ClassificationOutput`. In the training step, the computation of gradients is straightforward, necessitating a simple invocation of `backward()`on the loss. 

Note : Contrary to PyTorch, gradients are not stored alongside each tensor parameter, but are rather returned by the backward pass, as such: `let gradients = loss.backward()`. 

The gradient of a parameter can be obtained with the grad function: `let grad = tensor.grad(&gradients);`. Altough it is not necessary when using the learner struct adn the optimizers, it can prove to be quite useful when debugging or writing custom training loops. One difference between training and validation is the device mode: training uses an *autodiff-enabled* device so `backward` records and traverses the graph. 

Establishing the practical training configuration : 
```rust
#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig, 
    pub optimizer: AdamConfig, 
    #[config(default=10)]
    pub num_epochs: usize, 
    #[config(default=64)]
    pub batch_size: usize, 
    #[config(default=4)]
    pub num_workers: usize, 
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64 
} 

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_file(PathBuf::from(artifact_dir).join("experiment.log")).ok(); 
    std::fs::create_dir_all(artifact_dir).ok();
} 

pub fn train(artifact_dir: &str, config: TrainingConfig, device: impl Into<Device>) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully"); 

    let device = device.into(); 
    device.seed(config.seed); 
    let autodiff_device = device.autodiff(); 

    let batcher = MnistBatcher::default(); 

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train()); 

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test()); 

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .with_default_checkpointers()
        .num_epochs(config.num_epochs)
        .summary(); 

    let model = config.model.init(&autodiff_device); 
    let result = training.launch(Learner::new(
        model, config.optimizer.init(), config.learning_rate
    )); 

    result  
        .model 
        .into_record()
        .save(format!("{artifact_dir}/model"))
        .expect("Trained model should be saved successfully");

}
``` 

It is a good practice to use the `Config` derive to create the experiment configuration. In the `train` function, the first thing is making sure the `artifact_dir` exist, using the standard rust library for file manipulation. All checkpoints, logging, and metrics will be stored under this directory. Initialize the dataloaders using the previously created batcher. The selected runtime device is cloned and switched to autodiff mode before the model is initilized. 

Next, create a supervised training runner with the dataloaders for training and validation and register the accuracy and loss metrics on both training and validation steps. Enable checkpointing with `with_default_checkpointers()`, which periodically saves the model, optimizer, and learning rate scheduler state to burnpack files under the experiment directory so training can be resumed. 

**For the sake of simplicity in this example, this one is employ the test set as the validation set; however, isn't recommended this practice for actual usage.** 

The learner containing the model, the optimizer and the learning rate. Notably, the third argument of the learner's `new` function should actually be a learning rate *scheduler*. When provided with a float as in our example, it is automatically transformed into a *constant* learning rate scheduler. 

Once the learner and supervised training instance are created, we can call `training.launch` and provide the learner. 

Then, finally, the trained model is returned by the `launch` method. The trained weights are then saved by taking a record with `into_record()` and calling `save`, which writes a burnpack (`.bpk`) file. A record holds plain tensor data, so any backend, regardless of precision, can load recorded weights of any kind. 

