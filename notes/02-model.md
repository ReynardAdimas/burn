# Model 

In the `Cargo.toml` file, add the `burn` depedency with `train`, `vision`, and `flex` features. The default features automatically enable `tui` (for the dashboard), `fusion` and `autotune` for wgpu, among other defaults. 

```rust
[package]
name = "my_burn_app"
version = "0.1.0"
edition = "2024"

[dependencies]
burn = { version = "0.22.0", features = ["flex", "train", "vision"] }
```

The goal will be to create a basic convolutional neural network used for image classification. We will keep the model simple by using 2 conv layers followed by 2 linear layers, some pooling and ReLU activations. We will also use dropout to improve training performance. 

```rust
use burn::{
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig, Relu, 
        conv::{Conv2d, Conv2dConfig}, 
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig}
    }, 
    prelude::*
}; 

#[derive(Debug, Module)]
pub struct Model {
    conv1: Conv2d, 
    conv2: Conv2d, 
    pool: AdaptiveAvgPool2d,
    activation: Relu,
    linear1: Linear, 
    linear2: Linear, 
    dropout: Dropout, 
} 
```

These are two major things going on: 
1. You can create a deep learning module with the `#[derive(Module)]` attribute on top of a struct. This will generate the necessary code so that the struct implements the `Module` trait. This trait will make your module both trainable and (de)seriaizable while adding related functionalities. Like othe attributes often used in Rust, such as `Clone` , `PartialEq`, or `Debug`, each field within the struct must also implement the `Module` trait. 

2. The model and its layers don't have a backend type parameter. Tensors carry a runtime `Device`, and Burn routes their operations through its Tensor --> Bridge --> Dispatch --> Backend execution stack. This keeps models portable accross backend without exposing backend generics in the user API. The device used to initialize the parameters determines where the model starts executing. 


Next, we need to instantiate the model for training.
```rust 
#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize, 
    hidden_size: usize, 
    #[config(default = "0.5")]
    dropout: f64,
} 

impl ModelConfig {
    pub fn init(&self, device: &Device) -> Model {
        Model { 
            conv1: Conv2dConfig::new([1,8], [3,3]).init(device), 
            conv2: Conv2dConfig::new([8,16], [3,3]).init(device), 
            pool: AdaptiveAvgPool2dConfig::new([8,8]).init(), 
            activation: Relu::new(), 
            linear1: LinearConfig::new(16*8*8, self.hidden_size).init(device), 
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device), 
            dropout: DropoutConfig::new(self.dropout).init(), 
        }
    }
}
```

View the model configuration by printing the model instance: 
```rust
fn main() {
    let device = Device::flex(); 
    let model = ModelConfig::new(10, 512).init(&device); 

    println!("{model}");

}
``` 

When creating a custom neural network module, it is ofted a good idea to create a config alongside the model struct. This allows us to define default value for network. 

The beneift of `Config` attribute is that it makes the configuration serializable, enabling to painlessly save model hyperparameters, enchancing experimentation process. 

The first implementation block is related to the initialization method. As we can see, all fields are set using the configuration of the corresponding neural network's underlying module. In this spesific case, we have chosen to expand the tensor channel from 1 to 8 with the first layer, then from 8 to 16 with the second layer, using a kernel size of 3 on all dimensions. We also use the adaptive average pooling module to reduce the dimensionality of the images to an 8 by 8 matrix, which we will flatten in the forward pass to have a 1024(16x8x8) resulting tensor. 

Defined forward pass : 
```rust
impl Model {
    /// # Shapes 
    /// - Images [batch_size, height, width]
    /// - Output [batch_size, class_prob]
    pub fn forward(&self, images: Tensor<3>) -> Tensor<2> {
        let [batch_size, height, width] = images.dims(); 

        // Create a channel 
        let x = images.reshape([batch_size, 1, height, width]); 

        let x = self.conv1.forward(x); // [batch_size, 8, _, _]
        let x = self.dropout.forward(x); 
        let x = self.conv2.forward(x); // [batch_size, 16, _, _]
        let x = self.dropout.forward(x); 
        let x = self.activation.forward(x); 

        let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
        let x = x.reshape([batch_size,16*8*8]); 
        let x = self.linear1.forward(x); 
        let x = self.dropout.forward(x); 
        let x = self.activation.forward(x); 
        
        self.linear2.forward(x) // [batch_size, num_classes]
    }
} 
``` 

For former PyTorch users, this might feel very intuitive, as each module is directly incorporated into the code using an eager API. Most of the neural network modules already built with Burn use the `forward` nomenclature, simply because it is standard in the field.