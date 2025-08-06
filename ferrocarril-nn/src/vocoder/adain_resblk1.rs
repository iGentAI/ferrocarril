//! AdaINResBlock1 - Residual Block with AdaIN and Snake activation

use crate::{Parameter, Forward, conv::Conv1d, adain::AdaIN1d};
use ferrocarril_core::tensor::Tensor;
use super::{UpSample1d, UpsampleType};

#[cfg(feature = "weights")]
use ferrocarril_core::weights_binary::BinaryWeightLoader;
#[cfg(feature = "weights")]
use ferrocarril_core::LoadWeightsBinary;
#[cfg(feature = "weights")]
use ferrocarril_core::FerroError;

/// Snake1D activation function
#[inline]
pub fn snake1d(x: f32, alpha: f32) -> f32 {
    x + (1.0 / alpha) * ((alpha * x).sin().powi(2))
}

/// AdaINResBlock1 with Snake1D activation and optional upsampling
pub struct AdaINResBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    adain1: Vec<AdaIN1d>,
    adain2: Vec<AdaIN1d>,
    alpha1: Vec<Parameter>,
    alpha2: Vec<Parameter>,
    upsample: Option<UpSample1d>,
    conv1x1: Option<Conv1d>,  // Added for learned residual path when upsampling changes shape
    learned_sc: bool,         // Flag to indicate if we need a learned shortcut
}

impl AdaINResBlock1 {
    pub fn new(
        channels: usize,
        kernel_size: usize, 
        dilation: Vec<usize>,
        style_dim: usize
    ) -> Self {
        Self::with_upsample(channels, kernel_size, dilation, style_dim, None)
    }
    
    pub fn with_upsample(
        channels: usize,
        kernel_size: usize, 
        dilation: Vec<usize>,
        style_dim: usize,
        upsample_type: Option<UpsampleType>
    ) -> Self {
        assert_eq!(dilation.len(), 3, "Need exactly 3 dilation values");
        
        let mut convs1 = Vec::with_capacity(dilation.len());
        let mut convs2 = Vec::with_capacity(dilation.len());
        let mut adain1 = Vec::with_capacity(dilation.len());
        let mut adain2 = Vec::with_capacity(dilation.len());
        let mut alpha1 = Vec::with_capacity(dilation.len());
        let mut alpha2 = Vec::with_capacity(dilation.len());
        
        for &d in &dilation {
            let padding = get_padding(kernel_size, d);
            
            // First convolution with dilation
            convs1.push(Conv1d::new(
                channels, 
                channels, 
                kernel_size,
                1, // stride
                padding,
                d, // dilation
                1, // groups
                true // bias
            ));
            
            // Second convolution without dilation
            convs2.push(Conv1d::new(
                channels,
                channels,
                kernel_size,
                1, // stride
                get_padding(kernel_size, 1),
                1, // dilation
                1, // groups
                true // bias
            ));
            
            // AdaIN layers
            adain1.push(AdaIN1d::new(channels, style_dim));
            adain2.push(AdaIN1d::new(channels, style_dim));
            
            // Alpha parameters for Snake1D with shape [1, channels, 1] for broadcasting
            let alpha1_tensor = Tensor::from_data(vec![1.0; 1 * channels * 1], vec![1, channels, 1]);
            let alpha2_tensor = Tensor::from_data(vec![1.0; 1 * channels * 1], vec![1, channels, 1]);
            
            alpha1.push(Parameter::new(alpha1_tensor));
            alpha2.push(Parameter::new(alpha2_tensor));
        }
        
        // Determine if we need a learned shortcut for the residual connection
        let learned_sc = match upsample_type {
            Some(UpsampleType::Nearest) => true, // Always need learned shortcut when upsampling
            _ => false,
        };
        
        // Create appropriate conv1x1 for learned shortcut if needed
        let conv1x1 = if learned_sc {
            Some(Conv1d::new(
                channels,  // in_channels
                channels,  // out_channels
                1,         // kernel_size
                1,         // stride
                0,         // padding
                1,         // dilation
                1,         // groups
                false,     // bias (set to false as in Kokoro)
            ))
        } else {
            None
        };
        
        let upsample = upsample_type.map(UpSample1d::new);
        
        Self {
            convs1,
            convs2,
            adain1,
            adain2,
            alpha1,
            alpha2,
            upsample,
            conv1x1,
            learned_sc,
        }
    }
    
    /// Get the upsampling type for this block
    pub fn upsample_type(&self) -> UpsampleType {
        match &self.upsample {
            Some(up) => up.layer_type.clone(),
            None => UpsampleType::None,
        }
    }
    
    /// Process the shortcut (residual) path
    /// This is equivalent to Kokoro's _shortcut method
    fn _shortcut(&self, x: &Tensor<f32>) -> Tensor<f32> {
        let mut result = x.clone();
        
        // Apply upsampling to the shortcut path if needed
        if let Some(ref up) = self.upsample {
            result = up.forward(&result);
            
            // Verify upsampling behavior matches Kokoro's implementation
            // In Kokoro: time dimension should be doubled after upsampling
            assert_eq!(result.shape()[2], x.shape()[2] * 2,
                "Upsampling should double the time dimension. Expected {}, got {}",
                x.shape()[2] * 2, result.shape()[2]);
        }
        
        // Apply conv1x1 if we have a learned shortcut (exactly like Kokoro)
        if self.learned_sc {
            if let Some(ref conv) = self.conv1x1 {
                result = conv.forward(&result);
                
                // In Kokoro, after conv1x1 with learned_sc, the channel dimension should change
                // if upsampling was performed but shape[2] should be preserved
                if let Some(ref _up) = self.upsample {
                    assert_eq!(result.shape()[0], x.shape()[0],
                        "Batch dimension should be preserved after conv1x1");
                    // Don't assert on channel dimension equality - learned_sc means they can differ
                    assert_eq!(result.shape()[2], x.shape()[2] * 2,
                        "Time dimension should remain doubled after conv1x1");
                }
            }
        }
        
        result
    }
    
    /// Process the residual block path
    /// This is equivalent to Kokoro's _residual method
    fn _residual(&self, x: &Tensor<f32>, s: &Tensor<f32>) -> Tensor<f32> {
        let mut result = x.clone();
        
        for i in 0..self.convs1.len() {
            // Process through AdaIN and Snake1D (matching Kokoro exactly)
            let mut xt = self.adain1[i].forward(&result, s);
            
            // Apply Snake1D activation with PyTorch broadcasting pattern
            let alpha1_param = &self.alpha1[i];
            
            // Verify alpha parameter has correct shape for broadcasting
            assert_eq!(alpha1_param.data().shape(), &[1, xt.shape()[1], 1],
                "Alpha1 parameter must have shape [1, channels, 1] for broadcasting, got: {:?}",
                alpha1_param.data().shape());
            let shape = xt.shape().to_vec();
            let mut xt_data = xt.data().to_vec();
            let (batch, channels, time) = (shape[0], shape[1], shape[2]);
            
            for b in 0..batch {
                for c in 0..channels {
                    for t in 0..time {
                        let idx = b * channels * time + c * time + t;
                        let alpha = alpha1_param.data()[&[0, c, 0]];
                        let x_val = xt_data[idx];
                        xt_data[idx] = x_val + (1.0 / alpha) * ((alpha * x_val).sin().powi(2));
                    }
                }
            }
            xt = Tensor::from_data(xt_data, shape);
            
            // Apply upsampling if this is the first iteration and we have upsampling
            // This matches Kokoro: if i==0 and self.upsample is not None: xt = self.upsample(xt)
            if i == 0 && self.upsample.is_some() {
                if let Some(ref up) = self.upsample {
                    xt = up.forward(&xt);
                    
                    // Verify upsampling behavior
                    assert_eq!(xt.shape()[2], x.shape()[2] * 2,
                        "Upsampling should double the time dimension in residual path");
                }
            }
            
            // First convolution (matches Kokoro: xt = c1(xt))
            xt = self.convs1[i].forward(&xt);
            
            // Second AdaIN (matches Kokoro: xt = n2(xt, s))
            xt = self.adain2[i].forward(&xt, s);
            
            // Apply second Snake1D activation (matches Kokoro) with PyTorch broadcasting
            let alpha2_param = &self.alpha2[i];
            
            // Verify alpha parameter has correct shape for broadcasting
            assert_eq!(alpha2_param.data().shape(), &[1, xt.shape()[1], 1],
                "Alpha2 parameter must have shape [1, channels, 1] for broadcasting, got: {:?}",
                alpha2_param.data().shape());
            
            let shape = xt.shape().to_vec();
            let mut xt_data = xt.data().to_vec();
            let (batch, channels, time) = (shape[0], shape[1], shape[2]);
            
            for b in 0..batch {
                for c in 0..channels {
                    for t in 0..time {
                        let idx = b * channels * time + c * time + t;
                        let alpha = alpha2_param.data()[&[0, c, 0]];
                        let x_val = xt_data[idx];
                        xt_data[idx] = x_val + (1.0 / alpha) * ((alpha * x_val).sin().powi(2));
                    }
                }
            }
            xt = Tensor::from_data(xt_data, shape);
            
            // Second convolution (matches Kokoro: xt = c2(xt))
            xt = self.convs2[i].forward(&xt);
            
            // Update result for next iteration
            result = xt;
        }
        
        result
    }
    
    pub fn forward(&self, x: &Tensor<f32>, s: &Tensor<f32>) -> Tensor<f32> {
        // STRICT: Process residual and shortcut paths exactly like PyTorch
        let out = self._residual(x, s);
        let shortcut = self._shortcut(x);
        
        // STRICT: Shapes MUST match exactly or the architecture is wrong
        assert_eq!(out.shape(), shortcut.shape(),
            "CRITICAL ARCHITECTURAL ERROR: Residual path {:?} and shortcut path {:?} shapes don't match. \
            This indicates a fundamental bug in the channel transformation logic. \
            NO SILENT ADAPTATIONS ALLOWED - FIX THE ARCHITECTURE.",
            out.shape(), shortcut.shape());
        
        // Add paths with normalization
        let norm_factor = 1.0 / (2.0f32).sqrt();
        let mut result_data = vec![0.0; out.data().len()];
        
        for i in 0..out.data().len() {
            result_data[i] = (out.data()[i] + shortcut.data()[i]) * norm_factor;
        }
        
        Tensor::from_data(result_data, out.shape().to_vec())
    }
}

/// Helper function to calculate padding based on kernel size and dilation
fn get_padding(kernel_size: usize, dilation: usize) -> usize {
    (kernel_size * dilation - dilation) / 2
}

#[cfg(feature = "weights")]
impl LoadWeightsBinary for AdaINResBlock1 {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str
    ) -> Result<(), FerroError> {
        println!("Loading AdaINResBlock1 weights for {}.{} (Generator-style)", component, prefix);
        
        // PyTorch Generator uses convs1/convs2 arrays with multiple dilation layers
        // convs1.0, convs1.1, convs1.2 (3 layers) and convs2.0, convs2.1, convs2.2
        
        // Load convs1 layers (multiple dilation convolutions)
        for (i, conv) in self.convs1.iter_mut().enumerate() {
            let conv_prefix = format!("{}.convs1.{}", prefix, i);
            conv.load_weights_binary(loader, component, &conv_prefix)
                .map_err(|e| FerroError::new(format!("CRITICAL: AdaINResBlock1 convs1.{} loading FAILED: {}", i, e)))?;
        }
        
        // Load convs2 layers (fixed dilation=1 convolutions)
        for (i, conv) in self.convs2.iter_mut().enumerate() {
            let conv_prefix = format!("{}.convs2.{}", prefix, i);
            conv.load_weights_binary(loader, component, &conv_prefix)
                .map_err(|e| FerroError::new(format!("CRITICAL: AdaINResBlock1 convs2.{} loading FAILED: {}", i, e)))?;
        }
        
        // Load conv1x1 shortcut weights if we have a learned shortcut
        if self.learned_sc {
            if let Some(ref mut conv1x1) = self.conv1x1 {
                let conv_prefix = format!("{}.conv1x1", prefix);
                conv1x1.load_weights_binary(loader, component, &conv_prefix)
                    .map_err(|e| FerroError::new(format!("CRITICAL: AdaINResBlock1 conv1x1 loading FAILED: {}", e)))?;
            }
        }
        
        // Load AdaIN1 layers (adain1.0, adain1.1, adain1.2)
        for (i, adain) in self.adain1.iter_mut().enumerate() {
            let adain_prefix = format!("{}.adain1.{}", prefix, i);
            adain.load_weights_binary(loader, component, &adain_prefix)
                .map_err(|e| FerroError::new(format!("CRITICAL: AdaINResBlock1 adain1.{} loading FAILED: {}", i, e)))?;
        }
        
        // Load AdaIN2 layers (adain2.0, adain2.1, adain2.2)
        for (i, adain) in self.adain2.iter_mut().enumerate() {
            let adain_prefix = format!("{}.adain2.{}", prefix, i);
            adain.load_weights_binary(loader, component, &adain_prefix)
                .map_err(|e| FerroError::new(format!("CRITICAL: AdaINResBlock1 adain2.{} loading FAILED: {}", i, e)))?;
        }
        
        // Load alpha parameters (alpha1.0, alpha1.1, alpha1.2)
        for (i, alpha) in self.alpha1.iter_mut().enumerate() {
            let alpha_path = format!("{}.alpha1.{}", prefix, i);
            let alpha_tensor = loader.load_component_parameter(component, &alpha_path)
                .map_err(|e| FerroError::new(format!("CRITICAL: AdaINResBlock1 alpha1.{} loading FAILED: {}", i, e)))?;
            *alpha = Parameter::new(alpha_tensor);
        }
        
        // Load alpha2 parameters (alpha2.0, alpha2.1, alpha2.2)
        for (i, alpha) in self.alpha2.iter_mut().enumerate() {
            let alpha_path = format!("{}.alpha2.{}", prefix, i);
            let alpha_tensor = loader.load_component_parameter(component, &alpha_path)
                .map_err(|e| FerroError::new(format!("CRITICAL: AdaINResBlock1 alpha2.{} loading FAILED: {}", i, e)))?;
            *alpha = Parameter::new(alpha_tensor);
        }
        
        println!("✅ AdaINResBlock1: All Generator-style weights loaded successfully");
        Ok(())
    }
}

pub struct AdainResBlk1d {
    conv1: Conv1d,
    conv2: Conv1d,
    norm1: AdaIN1d,
    norm2: AdaIN1d,
    conv1x1: Option<Conv1d>,
    upsample: Option<UpSample1d>,
    learned_sc: bool,
    dim_in: usize,
    dim_out: usize,
}

impl AdainResBlk1d {
    pub fn new(
        dim_in: usize,
        dim_out: usize,
        style_dim: usize,
        upsample_type: Option<UpsampleType>,
    ) -> Self {
        let learned_sc = (dim_in != dim_out) || upsample_type.is_some();
        
        let conv1 = Conv1d::new(
            dim_in,
            dim_out,
            3,
            1,
            1,
            1,
            1,
            true,
        );
        
        let conv2 = Conv1d::new(
            dim_out,
            dim_out,
            3,
            1,
            1,
            1,
            1,
            true,
        );
        
        let norm1 = AdaIN1d::new(dim_in, style_dim);
        let norm2 = AdaIN1d::new(dim_out, style_dim);
        
        let conv1x1 = if learned_sc {
            Some(Conv1d::new(
                dim_in,
                dim_out,
                1,
                1,
                0,
                1,
                1,
                false,
            ))
        } else {
            None
        };
        
        let upsample = upsample_type.map(UpSample1d::new);
        
        Self {
            conv1,
            conv2,
            norm1,
            norm2,
            conv1x1,
            upsample,
            learned_sc,
            dim_in,
            dim_out,
        }
    }
    
    pub fn forward(&self, x: &Tensor<f32>, s: &Tensor<f32>) -> Tensor<f32> {
        let mut xt = self.norm1.forward(x, s);
        xt = self.apply_snake_activation(xt);
        
        if let Some(ref up) = self.upsample {
            xt = up.forward(&xt);
        }
        
        xt = self.conv1.forward(&xt);
        xt = self.norm2.forward(&xt, s);
        xt = self.apply_snake_activation(xt);
        let xt = self.conv2.forward(&xt);
        
        let mut shortcut = x.clone();
        
        if let Some(ref up) = self.upsample {
            shortcut = up.forward(&shortcut);
        }
        
        if self.learned_sc {
            if let Some(ref conv) = self.conv1x1 {
                shortcut = conv.forward(&shortcut);
            }
        }
        
        self.combine_paths(&xt, &shortcut)
    }
    
    fn apply_snake_activation(&self, mut tensor: Tensor<f32>) -> Tensor<f32> {
        let alpha = 1.0f32;
        let mut data = tensor.data().to_vec();
        for val in data.iter_mut() {
            *val = snake1d(*val, alpha);
        }
        Tensor::from_data(data, tensor.shape().to_vec())
    }
    
    fn combine_paths(&self, residual: &Tensor<f32>, shortcut: &Tensor<f32>) -> Tensor<f32> {
        let norm_factor = 1.0 / (2.0f32).sqrt();
        
        if residual.shape() == shortcut.shape() {
            let mut result_data = vec![0.0; residual.data().len()];
            for i in 0..residual.data().len() {
                result_data[i] = (residual.data()[i] + shortcut.data()[i]) * norm_factor;
            }
            Tensor::from_data(result_data, residual.shape().to_vec())
        } else {
            panic!("AdainResBlk1d: Residual and shortcut paths have mismatched shapes: {:?} vs {:?}",
                   residual.shape(), shortcut.shape());
        }
    }
}

#[cfg(feature = "weights")]
impl LoadWeightsBinary for AdainResBlk1d {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str
    ) -> Result<(), FerroError> {
        println!("Loading AdainResBlk1d weights for {}.{} (dim_in={} → dim_out={})", 
                 component, prefix, self.dim_in, self.dim_out);
        
        // Load conv1 using weight normalization format
        let conv1_weight_g = loader.load_component_parameter(component, &format!("{}.conv1.weight_g", prefix))?;
        let conv1_weight_v = loader.load_component_parameter(component, &format!("{}.conv1.weight_v", prefix))?;
        
        let conv1_weight = Self::reconstruct_weight_from_norm_static(&conv1_weight_g, &conv1_weight_v)?;
        self.conv1.load_from_reconstructed_weight(&conv1_weight)?;
        
        if let Ok(conv1_bias) = loader.load_component_parameter(component, &format!("{}.conv1.bias", prefix)) {
            self.conv1.set_bias(&conv1_bias)?;
        }
        
        // Load conv2 using weight normalization format
        let conv2_weight_g = loader.load_component_parameter(component, &format!("{}.conv2.weight_g", prefix))?;
        let conv2_weight_v = loader.load_component_parameter(component, &format!("{}.conv2.weight_v", prefix))?;
        
        let conv2_weight = Self::reconstruct_weight_from_norm_static(&conv2_weight_g, &conv2_weight_v)?;
        self.conv2.load_from_reconstructed_weight(&conv2_weight)?;
        
        if let Ok(conv2_bias) = loader.load_component_parameter(component, &format!("{}.conv2.bias", prefix)) {
            self.conv2.set_bias(&conv2_bias)?;
        }
        
        // Load conv1x1 if exists (for learned shortcut)  
        if self.learned_sc {
            if let Some(ref mut conv1x1) = self.conv1x1 {
                if let Ok(conv1x1_weight_g) = loader.load_component_parameter(component, &format!("{}.conv1x1.weight_g", prefix)) {
                    if let Ok(conv1x1_weight_v) = loader.load_component_parameter(component, &format!("{}.conv1x1.weight_v", prefix)) {
                        let conv1x1_weight = Self::reconstruct_weight_from_norm_static(&conv1x1_weight_g, &conv1x1_weight_v)?;
                        conv1x1.load_from_reconstructed_weight(&conv1x1_weight)?;
                    }
                }
            }
        }
        
        // Load normalization layers
        self.norm1.load_weights_binary(loader, component, &format!("{}.norm1", prefix))?;
        self.norm2.load_weights_binary(loader, component, &format!("{}.norm2", prefix))?;
        
        println!("✅ AdainResBlk1d: All weights loaded successfully with weight normalization");
        Ok(())
    }
}

impl AdainResBlk1d {
    /// Static method to reconstruct regular weight from PyTorch weight normalization
    fn reconstruct_weight_from_norm_static(weight_g: &Tensor<f32>, weight_v: &Tensor<f32>) -> Result<Tensor<f32>, FerroError> {
        let (out_channels, in_channels, kernel_size) = (
            weight_v.shape()[0], 
            weight_v.shape()[1], 
            weight_v.shape()[2]
        );
        
        // Handle weight_g shape variations ([C] or [C, 1, 1])
        let g_data = if weight_g.shape().len() == 1 {
            weight_g.data().to_vec()
        } else if weight_g.shape().len() == 3 && weight_g.shape()[1] == 1 && weight_g.shape()[2] == 1 {
            // Extract from [C, 1, 1] format
            let mut g_1d = Vec::new();
            for c in 0..out_channels {
                g_1d.push(weight_g[&[c, 0, 0]]);
            }
            g_1d
        } else {
            return Err(FerroError::new(format!("Invalid weight_g shape: {:?}", weight_g.shape())));
        };
        
        // Reconstruct normalized weight: weight = weight_g * (weight_v / ||weight_v||)
        let mut normalized_weight = vec![0.0; out_channels * in_channels * kernel_size];
        
        for oc in 0..out_channels {
            // Calculate L2 norm for this output channel
            let mut norm_sq = 0.0;
            for ic in 0..in_channels {
                for k in 0..kernel_size {
                    let val = weight_v[&[oc, ic, k]];
                    norm_sq += val * val;
                }
            }
            let norm = norm_sq.sqrt() + 1e-8; // Add epsilon for stability
            let g_val = g_data[oc];
            
            // Apply weight normalization
            for ic in 0..in_channels {
                for k in 0..kernel_size {
                    let idx = oc * in_channels * kernel_size + ic * kernel_size + k;
                    normalized_weight[idx] = g_val * weight_v[&[oc, ic, k]] / norm;
                }
            }
        }
        
        Ok(Tensor::from_data(normalized_weight, vec![out_channels, in_channels, kernel_size]))
    }
}