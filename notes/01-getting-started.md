# Burn 
It's a deep leaning framework in the Rust. 

## Using Prelude
Burn comes with a variety of things in its core liblary. When creating a new model or using an existing one for infernce, you may need to import every single component you used, which could be a little verbose. 

To address it, a `prelude` module is provided, allowing you to easily import commonly used structs and macros as a group:
```rust
use burn::prelude::*;
```
which is equal to:
```rust
use burn::{
    config::Config, 
    module::Module, 
    nn, 
    tensor::{
        Bool, Device, DeviceIndex, DeviceKind, ElementConversion, Float, Int, Shape, SliceArg, Tensor, TensorData, cas::ToElement, s,
    }
} 
``` 

## Createa 2-dimensional tensor 
Creates a 2-dimensional tensor containing the values `2,3,4,5` 
```rust 
let tensor_1 = Tensor::<2>::from_data(
    [[2.,3.], [4.,5.]],
    &device
);
``` 
- `let`declares a variable. 
- `tensor_1` is the variable name. 
- Rust infers the variable's type from the expression on the right. 

Don't need to explicitily write the type here because Rust can infer it. 

```rust 
Tensor::<2>
``` 
The `::<2>` syntax is called **turbofish syntax**. It explicitly supplies a generic argument to a type or function. In Burn, `Tensor` has a const generic parameter that specifies the number of dimensions. 

For example: 

```rust 
Tensor::<1> // 1-dimensional tensor 
Tensor::<2> // 2-dimensional tensor 
Tensor::<2> // 3-dimensional tensor
``` 

In Rust, `::<...>` is useful when you want to specify generic arguments explicitly instead of relying on type inference. 