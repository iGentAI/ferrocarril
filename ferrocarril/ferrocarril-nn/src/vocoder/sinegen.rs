//! Sine wave generator for neural source-filter model 

use crate::{Parameter, Forward};
use ferrocarril_core::tensor::Tensor;
use rand::Rng;

/// SineGen - Sinusoidal signal generator for neural source-filter model
pub struct SineGen {
    samp_rate: usize,
    upsample_scale: usize,
    harmonic_num: usize,
    dim: usize,
    sine_amp: f32,
    noise_std: f32, 
    voiced_threshold: f32,
    flag_for_pulse: bool,
}

impl SineGen {
    pub fn new(
        samp_rate: usize, 
        upsample_scale: usize,
        harmonic_num: usize,
        sine_amp: f32,
        noise_std: f32,
        voiced_threshold: f32,
        flag_for_pulse: bool,
    ) -> Self {
        Self {
            samp_rate,
            upsample_scale,
            harmonic_num,
            dim: harmonic_num + 1,
            sine_amp,
            noise_std,
            voiced_threshold,
            flag_for_pulse,
        }
    }
    
    /// Convert F0 to unvoiced/voiced flag
    fn f02uv(&self, f0: &Tensor<f32>) -> Tensor<f32> {
        let mut uv = vec![0.0; f0.data().len()];
        
        for i in 0..f0.data().len() {
            uv[i] = if f0.data()[i] > self.voiced_threshold { 1.0 } else { 0.0 };
        }
        
        Tensor::from_data(uv, f0.shape().to_vec())
    }
    
    /// Convert F0 value to sine wave
    fn f02sine(&self, f0_values: &Tensor<f32>) -> Tensor<f32> {
        let shape = f0_values.shape();
        let batch_size = shape[0];
        
        // The input tensor can be [batch, time, 1] or [batch, time, dim]
        // Handle both cases properly
        let time = shape[1];
        let input_dim = if shape.len() > 2 { shape[2] } else { 1 };
        
        // Return early if dimensions are invalid
        if batch_size == 0 || time == 0 {
            return Tensor::from_data(vec![], vec![batch_size, time, self.dim]);
        }
        
        // Create a tensor with dimensions [batch, time, dim] containing F0 for each harmonic
        // First, create flat buffer to store data
        let mut f0_buf = vec![0.0; batch_size * time * self.dim];
        
        // Fill the buffer with F0 values for each harmonic
        for b in 0..batch_size {
            for t in 0..time {
                // Get the fundamental F0 value, handling different input dimensions
                let f0_value = if input_dim == 1 {
                    if shape.len() > 2 {
                        // [batch, time, 1]
                        let idx = b * time * input_dim + t * input_dim;
                        if idx < f0_values.data().len() {
                            f0_values.data()[idx]
                        } else {
                            0.0 // Default if out of bounds
                        }
                    } else {
                        // [batch, time]
                        let idx = b * time + t;
                        if idx < f0_values.data().len() {
                            f0_values.data()[idx]
                        } else {
                            0.0 // Default if out of bounds 
                        }
                    }
                } else {
                    // Multi-dimensional input - use the first channel
                    let idx = b * time * input_dim + t * input_dim;
                    if idx < f0_values.data().len() {
                        f0_values.data()[idx]
                    } else {
                        0.0
                    }
                };
                
                // Apply to each harmonic
                for i in 0..self.dim {
                    f0_buf[b * time * self.dim + t * self.dim + i] = f0_value * (i + 1) as f32;
                }
            }
        }
        
        // Create the f0_tensor with correct shape
        let f0_tensor = Tensor::from_data(f0_buf, vec![batch_size, time, self.dim]);
        
        // Convert to F0 in rad. The integer part n can be ignored
        // because 2 * torch.pi * n doesn't affect phase
        let mut rad_values = vec![0.0; batch_size * time * self.dim];
        for b in 0..batch_size {
            for t in 0..time {
                for d in 0..self.dim {
                    // Calculate harmonic frequencies
                    let idx = b * time * self.dim + t * self.dim + d;
                    if idx < f0_tensor.data().len() {
                        let f0 = f0_tensor.data()[idx] / self.samp_rate as f32;
                        rad_values[idx] = f0 % 1.0;
                    }
                }
            }
        }
        
        // Set initial phase with random value
        let mut rand_ini = vec![0.0; batch_size * self.dim];
        let mut rng = rand::thread_rng();
        for b in 0..batch_size {
            for d in 1..self.dim { // Skip first dimension (fundamental)
                // Random value between 0 and 1
                rand_ini[b * self.dim + d] = rng.gen::<f32>();
            }
            // Set first dimension to 0
            rand_ini[b * self.dim] = 0.0;
        }
        
        // Add initial phase to first time step
        for b in 0..batch_size {
            for d in 0..self.dim {
                rad_values[b * time * self.dim + 0 * self.dim + d] += rand_ini[b * self.dim + d];
            }
        }
        
        // Interpolate for the upsampling
        let rad_values_interp = if !self.flag_for_pulse {
            // Check for division by zero
            if self.upsample_scale == 0 {
                return Tensor::from_data(vec![], vec![batch_size, time, self.dim]);
            }
            
            let downsampled_time = time / self.upsample_scale;
            // Safety check
            if downsampled_time == 0 {
                return Tensor::from_data(rad_values, vec![batch_size, time, self.dim]);
            }
            
            // Downsample first
            let mut downsampled = vec![0.0; batch_size * downsampled_time * self.dim];
            for b in 0..batch_size {
                for t in 0..downsampled_time {
                    for d in 0..self.dim {
                        if t * self.upsample_scale < time {
                            downsampled[b * downsampled_time * self.dim + t * self.dim + d] = 
                                rad_values[b * time * self.dim + (t * self.upsample_scale) * self.dim + d];
                        }
                    }
                }
            }
            
            // Then interpolate back up
            let mut interp = vec![0.0; batch_size * time * self.dim];
            for b in 0..batch_size {
                for t in 0..time {
                    let src_idx = t / self.upsample_scale;
                    let src_pos = src_idx * self.upsample_scale;
                    let r = (t - src_pos) as f32 / self.upsample_scale as f32;
                    
                    for d in 0..self.dim {
                        // Bounds check for downsampled indices
                        if src_idx < downsampled_time {
                            let src_val = downsampled[b * downsampled_time * self.dim + src_idx * self.dim + d];
                            let next_idx = if src_idx + 1 < downsampled_time {
                                src_idx + 1
                            } else {
                                src_idx
                            };
                            
                            let next_val = downsampled[b * downsampled_time * self.dim + next_idx * self.dim + d];
                            interp[b * time * self.dim + t * self.dim + d] = src_val * (1.0 - r) + next_val * r;
                        }
                    }
                }
            }
            interp
        } else {
            // TODO: Implement pulse-train generation
            rad_values.clone()
        };
        
        // Calculate phase based on frequency
        let mut phase = vec![0.0; batch_size * time * self.dim];
        
        // Cumulative sum along time dimension
        for b in 0..batch_size {
            for d in 0..self.dim {
                let mut acc = 0.0;
                for t in 0..time {
                    let idx = b * time * self.dim + t * self.dim + d;
                    if idx < rad_values_interp.len() {
                        acc += rad_values_interp[idx];
                        phase[idx] = acc;
                    }
                }
            }
        }
        
        // Scale by 2*PI
        for i in 0..phase.len() {
            phase[i] *= 2.0 * std::f32::consts::PI;
        }
        
        // Convert to sine waves
        let mut sines = vec![0.0; batch_size * time * self.dim];
        for i in 0..phase.len() {
            sines[i] = phase[i].sin() * self.sine_amp;
        }
        
        Tensor::from_data(sines, vec![batch_size, time, self.dim])
    }
    
    pub fn forward(&self, f0: &Tensor<f32>) 
        -> (Tensor<f32>, Tensor<f32>, Tensor<f32>) {
        // f0: [batch_size, time, 1] or [batch_size, time]
        let shape = f0.shape();
        let batch_size = shape[0];
        let time = shape[1];
        let input_dim = if shape.len() > 2 { shape[2] } else { 1 };
        
        // Check for empty tensor case to avoid index out of bounds
        if batch_size == 0 || time == 0 {
            // Return empty tensors with correct shapes
            let empty_sine = Tensor::from_data(vec![], vec![batch_size, time, self.dim]);
            let empty_uv = Tensor::from_data(vec![], 
                           if shape.len() > 2 { vec![batch_size, time, 1] } else { vec![batch_size, time] });
            let empty_noise = Tensor::from_data(vec![], vec![batch_size, time, self.dim]);
            return (empty_sine, empty_uv, empty_noise);
        }
        
        // Create F0 for each harmonic overtone
        let mut f0_buf = vec![0.0; batch_size * time * self.dim];
        
        for b in 0..batch_size {
            for t in 0..time {
                // Get the fundamental F0 value, handling different input dimensions
                let f0_value = if input_dim == 1 {
                    if shape.len() > 2 {
                        // [batch, time, 1]
                        let idx = b * time * input_dim + t * input_dim;
                        if idx < f0.data().len() {
                            f0.data()[idx]
                        } else {
                            0.0 // Default if out of bounds
                        }
                    } else {
                        // [batch, time]
                        let idx = b * time + t;
                        if idx < f0.data().len() {
                            f0.data()[idx]
                        } else {
                            0.0 // Default if out of bounds
                        }
                    }
                } else {
                    // Should not happen with normal F0 input, but handle anyway
                    let idx = b * time * input_dim + t * input_dim;
                    if idx < f0.data().len() {
                        f0.data()[idx]
                    } else {
                        0.0
                    }
                };
                
                // For each harmonic
                for i in 0..self.harmonic_num + 1 {
                    f0_buf[b * time * self.dim + t * self.dim + i] = f0_value * (i + 1) as f32;
                }
            }
        }
        
        let f0_tensor = Tensor::from_data(f0_buf, vec![batch_size, time, self.dim]);
        
        // Generate sine waves
        let sine_waves = self.f02sine(&f0_tensor);
        
        // Generate UV signal
        let uv = self.f02uv(f0);
        
        // Generate noise
        let mut rng = rand::thread_rng();
        let mut noise = vec![0.0; batch_size * time * self.dim];
        for b in 0..batch_size {
            for t in 0..time {
                for d in 0..self.dim {
                    // Get the UV value for this position, handling different input shapes
                    let uv_value = if shape.len() > 2 {
                        // [batch, time, 1] tensor
                        let idx = b * time * 1 + t * 1; // Access the first (only) channel
                        if idx < uv.data().len() {
                            uv.data()[idx]
                        } else {
                            0.0
                        }
                    } else {
                        // [batch, time] tensor
                        let idx = b * time + t;
                        if idx < uv.data().len() {
                            uv.data()[idx]
                        } else {
                            0.0
                        }
                    };
                    
                    // Random noise with proper amplitude
                    let noise_amp = if uv_value > 0.0 {
                        self.noise_std
                    } else {
                        self.sine_amp / 3.0
                    };
                    noise[b * time * self.dim + t * self.dim + d] = (rng.gen::<f32>() * 2.0 - 1.0) * noise_amp;
                }
            }
        }
        
        let noise_tensor = Tensor::from_data(noise, vec![batch_size, time, self.dim]);
        
        // Set unvoiced part to 0 by uv and add noise
        let mut result = vec![0.0; batch_size * time * self.dim];
        for b in 0..batch_size {
            for t in 0..time {
                // Get the UV value for this position, handling different input shapes
                let uv_value = if shape.len() > 2 {
                    // [batch, time, 1] tensor
                    let idx = b * time * 1 + t * 1; // Access the first (only) channel
                    if idx < uv.data().len() {
                        uv.data()[idx]
                    } else {
                        0.0
                    }
                } else {
                    // [batch, time] tensor
                    let idx = b * time + t;
                    if idx < uv.data().len() {
                        uv.data()[idx]
                    } else {
                        0.0
                    }
                };
                
                for d in 0..self.dim {
                    let idx = b * time * self.dim + t * self.dim + d;
                    if idx < sine_waves.data().len() && idx < noise_tensor.data().len() {
                        result[idx] = sine_waves.data()[idx] * uv_value + noise_tensor.data()[idx];
                    }
                }
            }
        }
        
        let sine_with_noise = Tensor::from_data(result, vec![batch_size, time, self.dim]);
        
        (sine_with_noise, uv, noise_tensor)
    }
}

// Add new method for weight loading
#[cfg(feature = "weights")]
impl SineGen {
    // This method is just for conformity with the weight loading system
    // SineGen doesn't actually have learnable parameters to load
    pub fn sinegen_mut(&mut self) -> &mut Self {
        self
    }
}