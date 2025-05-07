//! 1D convolution implementation

use crate::{Parameter, Forward};
use ferrocarril_core::tensor::Tensor;

pub struct Conv1d {
    weight: Parameter,
    bias: Option<Parameter>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    in_channels: usize,
    out_channels: usize,
}

impl Conv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Self {
        assert!(in_channels % groups == 0, "in_channels must be divisible by groups");
        assert!(out_channels % groups == 0, "out_channels must be divisible by groups");
        
        let weight_shape = vec![out_channels, in_channels / groups, kernel_size];
        let weight = Parameter::new(Tensor::new(weight_shape));
        
        let bias = if bias {
            Some(Parameter::new(Tensor::new(vec![out_channels])))
        } else {
            None
        };
        
        Self {
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            in_channels,
            out_channels,
        }
    }
    
    fn conv1d(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let weight = self.weight.data();
        let input_shape = input.shape();
        let weight_shape = weight.shape();
        
        assert_eq!(input_shape.len(), 3, "Input must be 3D: [batch_size, channels, length]");
        let (batch_size, in_channels, in_length) = (input_shape[0], input_shape[1], input_shape[2]);
        
        assert_eq!(in_channels, self.in_channels, "Input channels mismatch");
        
        let kernel_size = weight_shape[2];
        let out_channels = self.out_channels;
        
        // Calculate output dimensions
        let dilated_kernel_size = (kernel_size - 1) * self.dilation + 1;
        let out_length = (in_length + 2 * self.padding - dilated_kernel_size) / self.stride + 1;
        
        let mut output = Tensor::new(vec![batch_size, out_channels, out_length]);
        
        // Perform convolution (naive implementation for MVP)
        for b in 0..batch_size {
            for oc in 0..out_channels {
                let group_id = oc / (out_channels / self.groups);
                let group_in_channels = in_channels / self.groups;
                let group_in_start = group_id * group_in_channels;
                
                for ol in 0..out_length {
                    let mut sum = 0.0;
                    
                    // Convolve over input channels in this group
                    for ic_rel in 0..group_in_channels {
                        let ic = group_in_start + ic_rel;
                        
                        // Convolve over kernel
                        for k in 0..kernel_size {
                            let il = ol * self.stride + k * self.dilation;
                            if il < self.padding || il >= in_length + self.padding {
                                continue;
                            }
                            let il_actual = il - self.padding;
                            if il_actual < in_length {
                                sum += input[&[b, ic, il_actual]] * weight[&[oc, ic_rel, k]];
                            }
                        }
                    }
                    
                    // Add bias if present
                    if let Some(ref bias) = self.bias {
                        sum += bias.data()[&[oc]];
                    }
                    
                    output[&[b, oc, ol]] = sum;
                }
            }
        }
        
        output
    }
}

impl Forward for Conv1d {
    type Output = Tensor<f32>;
    
    fn forward(&self, input: &Tensor<f32>) -> Self::Output {
        self.conv1d(input)
    }
}

/// Create a Conv1d layer with standard defaults
pub fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    dilation: usize,
) -> Conv1d {
    let padding = (kernel_size - 1) / 2 * dilation;
    Conv1d::new(
        in_channels,
        out_channels,
        kernel_size,
        1, // stride
        padding,
        dilation,
        1, // groups
        true, // bias
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_conv1d_forward() {
        let conv = Conv1d::new(3, 6, 3, 1, 1, 1, 1, true);
        let input = Tensor::new(vec![2, 3, 10]); // [batch_size, channels, length]
        let output = conv.forward(&input);
        
        assert_eq!(output.shape(), &[2, 6, 10]);
    }
}