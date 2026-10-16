# Backend 

Burn's default Cargo features do not include an execution backend. Applications select one explicitly, for example flex(CPU): 
```rust
burn = { version = "0.22", features = ["flex"] }
``` 

```rust
use crate::{model::ModelConfig, training::TrainingConfig};
use burn::{
    prelude::*,
    optim::AdamConfig,
};

fn main() {
    let device = Device::wgpu(Default::default());

    let artifact_dir = "target/guide";
    crate::training::train(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
}
``` 

The model itself remains backend-agnostic:tensor operations are dispatched according to their device. The traning function creates an autodiff-enabled device internally. 

Call the `train` function defined earlier with a directory for artifcts, the configuration of the model (the number of digit classes is 10 and the hidden dimension is 512), the optimizer configuration which in our case will be the default Adam configuration, and the device which can be obtained from the backend. 

Then train model with the command:
```rust
cargo run --release 
``` 



