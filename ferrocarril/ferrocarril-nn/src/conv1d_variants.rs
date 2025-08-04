//! Specialized Conv1d implementations for different Kokoro TTS components
//! 
//! PyTorch Kokoro uses Conv1d in 3 major patterns:
//! - TextEncoderConv1d: weight_norm CNN blocks for phoneme processing
//! - PredictorConv1d: F0/noise prediction convolutions
//! - DecoderConv1d: massive vocoder Conv1d complexity (190 weights)

use crate::Parameter;
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::{FerroError, LoadWeightsBinary};
use ferrocarril_core::weights_binary::BinaryWeightLoader;

/// TextEncoder Conv1d: Weight-normalized conv blocks for phoneme processing
/// 
/// Uses PyTorch's weight_norm which stores weights as weight_g (scale) and weight_v (direction)
#[derive(Debug)]
pub struct TextEncoderConv1d {
    weight_g: Parameter,     // [out_channels] - magnitude scaling
    weight_v: Parameter,     // [out_channels, in_channels, kernel_size] - direction
    bias: Option<Parameter>, // [out_channels]
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
}

impl TextEncoderConv1d {
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
        let weight_g = Parameter::new(Tensor::new(vec![out_channels]));
        let weight_v = Parameter::new(Tensor::new(vec![out_channels, in_channels / groups, kernel_size]));
        
        let bias = if bias {
            Some(Parameter::new(Tensor::new(vec![out_channels])))
        } else {
            None
        };
        
        Self {
            weight_g,
            weight_v,
            bias,
            stride,
            padding,
            dilation,
            groups,
            in_channels,
            out_channels,
            kernel_size,
        }
    }
    
    /// Set weight norm parameters from loaded weights
    pub fn set_weight_norm(&mut self, weight_g: &Tensor<f32>, weight_v: &Tensor<f32>) -> Result<(), FerroError> {
        // Validate shapes
        assert_eq!(weight_g.shape(), &[self.out_channels],
            "weight_g shape mismatch: expected [{}], got {:?}", self.out_channels, weight_g.shape());
        assert_eq!(weight_v.shape(), &[self.out_channels, self.in_channels / self.groups, self.kernel_size],
            "weight_v shape mismatch: expected [{}, {}, {}], got {:?}", 
            self.out_channels, self.in_channels / self.groups, self.kernel_size, weight_v.shape());
        
        self.weight_g = Parameter::new(weight_g.clone());
        self.weight_v = Parameter::new(weight_v.clone());
        Ok(())
    }
    
    /// Set bias parameter
    pub fn set_bias(&mut self, bias: &Tensor<f32>) -> Result<(), FerroError> {
        assert_eq!(bias.shape(), &[self.out_channels],
            "bias shape mismatch: expected [{}], got {:?}", self.out_channels, bias.shape());
        self.bias = Some(Parameter::new(bias.clone()));
        Ok(())
    }
    
    /// Forward pass with weight normalization applied
    pub fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        // Reconstruct weight from weight_g and weight_v (PyTorch weight_norm pattern)
        let weight_g = self.weight_g.data();
        let weight_v = self.weight_v.data();
        
        // Normalize weight_v by channel and scale by weight_g
        let mut normalized_weight = vec![0.0; self.out_channels * (self.in_channels / self.groups) * self.kernel_size];
        
        for oc in 0..self.out_channels {
            // Calculate L2 norm for this output channel
            let mut norm_sq = 0.0;
            for ic in 0..(self.in_channels / self.groups) {
                for k in 0..self.kernel_size {
                    let val = weight_v[&[oc, ic, k]];
                    norm_sq += val * val;
                }
            }
            let norm = norm_sq.sqrt();
            
            // Apply weight normalization: weight = weight_g * (weight_v / ||weight_v||)
            let g_val = weight_g[&[oc]];
            for ic in 0..(self.in_channels / self.groups) {
                for k in 0..self.kernel_size {
                    let idx = oc * (self.in_channels / self.groups) * self.kernel_size + ic * self.kernel_size + k;
                    normalized_weight[idx] = g_val * weight_v[&[oc, ic, k]] / norm;
                }
            }
        }
        
        let weight = Tensor::from_data(normalized_weight, vec![self.out_channels, self.in_channels / self.groups, self.kernel_size]);
        
        // Apply standard Conv1d with normalized weights
        self.conv1d(&weight, input)
    }
    
    /// Internal Conv1d implementation
    fn conv1d(&self, weight: &Tensor<f32>, input: &Tensor<f32>) -> Tensor<f32> {
        let input_shape = input.shape();
        assert_eq!(input_shape.len(), 3, "Input must be 3D: [batch_size, channels, length]");
        let (batch_size, in_channels, in_length) = (input_shape[0], input_shape[1], input_shape[2]);
        
        assert_eq!(in_channels, self.in_channels, "Input channels mismatch");
        
        // Calculate output dimensions
        let dilated_kernel_size = (self.kernel_size - 1) * self.dilation + 1;
        let out_length = (in_length + 2 * self.padding - dilated_kernel_size) / self.stride + 1;
        
        let mut output = Tensor::new(vec![batch_size, self.out_channels, out_length]);
        
        // Perform convolution
        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                let group_id = oc / (self.out_channels / self.groups);
                let group_in_channels = self.in_channels / self.groups;
                let group_in_start = group_id * group_in_channels;
                
                for ol in 0..out_length {
                    let mut sum = 0.0;
                    
                    for ic_rel in 0..group_in_channels {
                        let ic = group_in_start + ic_rel;
                        
                        for k in 0..self.kernel_size {
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

impl LoadWeightsBinary for TextEncoderConv1d {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        // Load weight_norm parameters (weight_g and weight_v pattern)
        let weight_g = loader.load_component_parameter(component, &format!("{}.weight_g", prefix))?;
        let weight_v = loader.load_component_parameter(component, &format!("{}.weight_v", prefix))?;
        
        self.set_weight_norm(&weight_g, &weight_v)?;
        
        // Load bias if it exists
        if let Ok(bias) = loader.load_component_parameter(component, &format!("{}.bias", prefix)) {
            self.set_bias(&bias)?;
        }
        
        Ok(())
    }
}

/// PredictorConv1d: F0 and noise prediction convolutions
/// 
/// Used in F0 and noise prediction blocks with specific kernel sizes for TTS prosody
#[derive(Debug)]
pub struct PredictorConv1d {
    weight_g: Parameter,
    weight_v: Parameter,
    bias: Option<Parameter>,
    stride: usize,
    padding: usize,
    kernel_size: usize,
    in_channels: usize,
    out_channels: usize,
}

impl PredictorConv1d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let padding = (kernel_size - 1) / 2; // Standard padding for same-size output
        
        Self {
            weight_g: Parameter::new(Tensor::new(vec![out_channels])),
            weight_v: Parameter::new(Tensor::new(vec![out_channels, in_channels, kernel_size])),
            bias: Some(Parameter::new(Tensor::new(vec![out_channels]))),
            stride: 1,
            padding,
            kernel_size,
            in_channels,
            out_channels,
        }
    }
    
    /// Forward pass optimized for prosody prediction
    pub fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        // Similar weight normalization as TextEncoderConv1d but optimized for predictor usage
        let weight_g = self.weight_g.data();
        let weight_v = self.weight_v.data();
        
        let mut normalized_weight = vec![0.0; self.out_channels * self.in_channels * self.kernel_size];
        
        for oc in 0..self.out_channels {
            let mut norm_sq = 0.0;
            for ic in 0..self.in_channels {
                for k in 0..self.kernel_size {
                    let val = weight_v[&[oc, ic, k]];
                    norm_sq += val * val;
                }
            }
            let norm = norm_sq.sqrt();
            let g_val = weight_g[&[oc]];
            
            for ic in 0..self.in_channels {
                for k in 0..self.kernel_size {
                    let idx = oc * self.in_channels * self.kernel_size + ic * self.kernel_size + k;
                    normalized_weight[idx] = g_val * weight_v[&[oc, ic, k]] / norm;
                }
            }
        }
        
        let weight = Tensor::from_data(normalized_weight, vec![self.out_channels, self.in_channels, self.kernel_size]);
        
        // Apply Conv1d (simplified for predictor usage)
        self.conv1d_predictor(&weight, input)
    }
    
    fn conv1d_predictor(&self, weight: &Tensor<f32>, input: &Tensor<f32>) -> Tensor<f32> {
        let input_shape = input.shape();
        let (batch_size, in_channels, in_length) = (input_shape[0], input_shape[1], input_shape[2]);
        
        let out_length = (in_length + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let mut output = Tensor::new(vec![batch_size, self.out_channels, out_length]);
        
        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for ol in 0..out_length {
                    let mut sum = 0.0;
                    
                    for ic in 0..self.in_channels {
                        for k in 0..self.kernel_size {
                            let il = ol * self.stride + k;
                            if il < self.padding || il >= in_length + self.padding {
                                continue;
                            }
                            let il_actual = il - self.padding;
                            if il_actual < in_length {
                                sum += input[&[b, ic, il_actual]] * weight[&[oc, ic, k]];
                            }
                        }
                    }
                    
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

impl LoadWeightsBinary for PredictorConv1d {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        // Predictor Conv1d uses weight_norm pattern
        let weight_g = loader.load_component_parameter(component, &format!("{}.weight_g", prefix))?;
        let weight_v = loader.load_component_parameter(component, &format!("{}.weight_v", prefix))?;
        
        // Update internal parameters
        self.weight_g = Parameter::new(weight_g);
        self.weight_v = Parameter::new(weight_v);
        
        // Load bias
        if let Ok(bias) = loader.load_component_parameter(component, &format!("{}.bias", prefix)) {
            self.bias = Some(Parameter::new(bias));
        }
        
        Ok(())
    }
}

/// DecoderConv1d: Vocoder convolutions with complex upsampling/downsampling
/// 
/// This handles the 190 decoder Conv1d weights with various configurations
/// Used in generator, encode/decode blocks, and noise processing
#[derive(Debug)]
pub struct DecoderConv1d {
    weight_g: Parameter,
    weight_v: Parameter,
    bias: Option<Parameter>,
    config: DecoderConv1dConfig,
}

#[derive(Debug)]
pub struct DecoderConv1dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub groups: usize,
    pub vocoder_type: VocoderConvType,
}

#[derive(Debug)]
pub enum VocoderConvType {
    Upsampling,      // Generator upsampling convolutions
    Resblock,        // Residual block convolutions  
    PostNet,         // Post-processing convolutions
    SourceModule,    // Source module convolutions
}

impl DecoderConv1d {
    pub fn new(config: DecoderConv1dConfig) -> Self {
        Self {
            weight_g: Parameter::new(Tensor::new(vec![config.out_channels])),
            weight_v: Parameter::new(Tensor::new(vec![config.out_channels, config.in_channels / config.groups, config.kernel_size])),
            bias: Some(Parameter::new(Tensor::new(vec![config.out_channels]))),
            config,
        }
    }
    
    /// Forward pass optimized for vocoder usage
    pub fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        // Weight normalization for decoder-specific patterns
        let weight_g = self.weight_g.data();
        let weight_v = self.weight_v.data();
        
        let mut normalized_weight = vec![0.0; self.config.out_channels * (self.config.in_channels / self.config.groups) * self.config.kernel_size];
        
        for oc in 0..self.config.out_channels {
            let mut norm_sq = 0.0;
            for ic in 0..(self.config.in_channels / self.config.groups) {
                for k in 0..self.config.kernel_size {
                    let val = weight_v[&[oc, ic, k]];
                    norm_sq += val * val;
                }
            }
            let norm = norm_sq.sqrt() + 1e-8; // Add small epsilon for stability
            let g_val = weight_g[&[oc]];
            
            for ic in 0..(self.config.in_channels / self.config.groups) {
                for k in 0..self.config.kernel_size {
                    let idx = oc * (self.config.in_channels / self.config.groups) * self.config.kernel_size + ic * self.config.kernel_size + k;
                    normalized_weight[idx] = g_val * weight_v[&[oc, ic, k]] / norm;
                }
            }
        }
        
        let weight = Tensor::from_data(
            normalized_weight, 
            vec![self.config.out_channels, self.config.in_channels / self.config.groups, self.config.kernel_size]
        );
        
        // Apply convolution with decoder-specific logic
        self.conv1d_decoder(&weight, input)
    }
    
    fn conv1d_decoder(&self, weight: &Tensor<f32>, input: &Tensor<f32>) -> Tensor<f32> {
        let input_shape = input.shape();
        let (batch_size, in_channels, in_length) = (input_shape[0], input_shape[1], input_shape[2]);
        
        let out_length = match self.config.vocoder_type {
            VocoderConvType::Upsampling => {
                // Upsampling convolutions may have different output length calculation
                (in_length * self.config.stride + self.config.kernel_size - 2 * self.config.padding - 1) / 1 + 1
            },
            _ => {
                // Standard calculation for other types
                (in_length + 2 * self.config.padding - self.config.kernel_size) / self.config.stride + 1
            }
        };
        
        let mut output = Tensor::new(vec![batch_size, self.config.out_channels, out_length]);
        
        // Standard convolution implementation (can be optimized per vocoder type)
        for b in 0..batch_size {
            for oc in 0..self.config.out_channels {
                let group_id = oc / (self.config.out_channels / self.config.groups);
                let group_in_channels = self.config.in_channels / self.config.groups;
                let group_in_start = group_id * group_in_channels;
                
                for ol in 0..out_length {
                    let mut sum = 0.0;
                    
                    for ic_rel in 0..group_in_channels {
                        let ic = group_in_start + ic_rel;
                        
                        for k in 0..self.config.kernel_size {
                            let il = ol * self.config.stride + k;
                            if il < self.config.padding || il >= in_length + self.config.padding {
                                continue;
                            }
                            let il_actual = il - self.config.padding;
                            if il_actual < in_length {
                                sum += input[&[b, ic, il_actual]] * weight[&[oc, ic_rel, k]];
                            }
                        }
                    }
                    
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

impl LoadWeightsBinary for DecoderConv1d {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        // Decoder Conv1d also uses weight_norm pattern
        let weight_g = loader.load_component_parameter(component, &format!("{}.weight_g", prefix))?;
        let weight_v = loader.load_component_parameter(component, &format!("{}.weight_v", prefix))?;
        
        self.weight_g = Parameter::new(weight_g);
        self.weight_v = Parameter::new(weight_v);
        
        // Load bias
        if let Ok(bias) = loader.load_component_parameter(component, &format!("{}.bias", prefix)) {
            self.bias = Some(Parameter::new(bias));
        }
        
        Ok(())
    }
}

/// Standard Conv1d for simple cases (1x1 projections, etc.)
#[derive(Debug)]
pub struct StandardConv1d {
    weight: Parameter,
    bias: Option<Parameter>,
    stride: usize,
    padding: usize,
    in_channels: usize,
    out_channels: usize, 
    kernel_size: usize,
}

impl StandardConv1d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self {
            weight: Parameter::new(Tensor::new(vec![out_channels, in_channels, kernel_size])),
            bias: Some(Parameter::new(Tensor::new(vec![out_channels]))),
            stride: 1,
            padding: 0,
            in_channels,
            out_channels,
            kernel_size,
        }
    }
    
    pub fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let input_shape = input.shape();
        let (batch_size, in_channels, in_length) = (input_shape[0], input_shape[1], input_shape[2]);
        
        let out_length = (in_length + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let mut output = Tensor::new(vec![batch_size, self.out_channels, out_length]);
        
        let weight = self.weight.data();
        
        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for ol in 0..out_length {
                    let mut sum = 0.0;
                    
                    for ic in 0..self.in_channels {
                        for k in 0..self.kernel_size {
                            let il = ol * self.stride + k;
                            if il < self.padding || il >= in_length + self.padding {
                                continue;
                            }
                            let il_actual = il - self.padding;
                            if il_actual < in_length {
                                sum += input[&[b, ic, il_actual]] * weight[&[oc, ic, k]];
                            }
                        }
                    }
                    
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

impl LoadWeightsBinary for StandardConv1d {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        // Standard weight/bias loading pattern
        let weight = loader.load_component_parameter(component, &format!("{}.weight", prefix))?;
        self.weight = Parameter::new(weight);
        
        if let Ok(bias) = loader.load_component_parameter(component, &format!("{}.bias", prefix)) {
            self.bias = Some(Parameter::new(bias));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_textencoder_conv1d() {
        let conv = TextEncoderConv1d::new(512, 512, 5, 1, 2, 1, 1, true);
        let input = Tensor::new(vec![2, 512, 10]);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[2, 512, 10]);
    }
    
    #[test]
    fn test_predictor_conv1d() {
        let conv = PredictorConv1d::new(256, 1, 1);  // F0/noise projection
        let input = Tensor::new(vec![1, 256, 50]);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[1, 1, 50]);
    }
}