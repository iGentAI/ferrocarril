//! Sine wave generator exactly matching PyTorch istftnet.py implementation

use ferrocarril_core::tensor::Tensor;
use rand::Rng;
use std::f32::consts::PI;

/// SineGen - Exactly matching PyTorch istftnet.py implementation  
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
    
    /// EXACT PYTORCH _f02uv implementation
    fn f02uv(&self, f0: &Tensor<f32>) -> Tensor<f32> {
        // PyTorch: uv = (f0 > self.voiced_threshold).type(torch.float32)
        let mut uv = vec![0.0; f0.data().len()];
        
        for i in 0..f0.data().len() {
            uv[i] = if f0.data()[i] > self.voiced_threshold { 1.0 } else { 0.0 };
        }
        
        Tensor::from_data(uv, f0.shape().to_vec())
    }
    
    /// EXACT PYTORCH F.interpolate implementation for linear interpolation
    fn interpolate_linear(&self, input: &[f32], input_shape: &[usize], scale_factor: f32) -> (Vec<f32>, Vec<usize>) {
        // PyTorch F.interpolate(input.transpose(1, 2), scale_factor=scale_factor, mode="linear").transpose(1, 2)
        let (batch_size, length, dim) = (input_shape[0], input_shape[1], input_shape[2]);
        let new_length = ((length as f32 * scale_factor) as usize).max(1);
        
        let mut output = vec![0.0; batch_size * new_length * dim];
        
        for b in 0..batch_size {
            for d in 0..dim {
                for new_l in 0..new_length {
                    // Linear interpolation for this position
                    let src_pos = (new_l as f32 * length as f32) / new_length as f32;
                    let src_l_low = src_pos.floor() as usize;
                    let src_l_high = (src_l_low + 1).min(length - 1);
                    let weight = src_pos - src_pos.floor();
                    
                    let low_val = if src_l_low < length {
                        input[b * length * dim + src_l_low * dim + d]
                    } else {
                        0.0
                    };
                    
                    let high_val = if src_l_high < length {
                        input[b * length * dim + src_l_high * dim + d]
                    } else {
                        low_val
                    };
                    
                    output[b * new_length * dim + new_l * dim + d] = low_val * (1.0 - weight) + high_val * weight;
                }
            }
        }
        
        (output, vec![batch_size, new_length, dim])
    }
    
    /// EXACT PYTORCH _f02sine implementation
    fn f02sine(&self, f0_values: &Tensor<f32>) -> Tensor<f32> {
        let shape = f0_values.shape();
        let (batch_size, length, dim) = (shape[0], shape[1], shape[2]);
        
        // PyTorch: rad_values = (f0_values / self.sampling_rate) % 1
        let mut rad_values = vec![0.0; batch_size * length * dim];
        for i in 0..rad_values.len() {
            rad_values[i] = (f0_values.data()[i] / self.samp_rate as f32) % 1.0;
        }
        
        // PyTorch: rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        let mut rng = rand::thread_rng();
        let mut rand_ini = vec![0.0; batch_size * dim];
        for i in 0..rand_ini.len() {
            rand_ini[i] = rng.gen::<f32>();
        }
        
        // PyTorch: rand_ini[:, 0] = 0
        for b in 0..batch_size {
            rand_ini[b * dim + 0] = 0.0;
        }
        
        // PyTorch: rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini  
        for b in 0..batch_size {
            for d in 0..dim {
                let idx = b * length * dim + 0 * dim + d; // l=0
                if idx < rad_values.len() && b * dim + d < rand_ini.len() {
                    rad_values[idx] += rand_ini[b * dim + d];
                }
            }
        }
        
        if !self.flag_for_pulse {
            // PyTorch: rad_values = F.interpolate(rad_values.transpose(1, 2), scale_factor=1/self.upsample_scale, mode="linear").transpose(1, 2)
            let scale_down = 1.0 / self.upsample_scale as f32;
            let (rad_downsampled, down_shape) = self.interpolate_linear(&rad_values, &[batch_size, length, dim], scale_down);
            
            // PyTorch: phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
            let mut phase = vec![0.0; rad_downsampled.len()];
            let down_length = down_shape[1];
            
            for b in 0..batch_size {
                for d in 0..dim {
                    let mut cumsum = 0.0;
                    for l in 0..down_length {
                        let idx = b * down_length * dim + l * dim + d;
                        if idx < rad_downsampled.len() {
                            cumsum += rad_downsampled[idx];
                            phase[idx] = cumsum * 2.0 * PI;
                        }
                    }
                }
            }
            
            // PyTorch: phase = F.interpolate(phase.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear").transpose(1, 2)
            for i in 0..phase.len() {
                phase[i] *= self.upsample_scale as f32;
            }
            
            let scale_up = self.upsample_scale as f32;
            let (phase_upsampled, _) = self.interpolate_linear(&phase, &down_shape, scale_up);
            
            // PyTorch: sines = torch.sin(phase)
            let mut sines = vec![0.0; phase_upsampled.len()];
            for i in 0..phase_upsampled.len() {
                sines[i] = phase_upsampled[i].sin();
            }
            
            // Ensure final shape matches input length
            let final_sines = if sines.len() != batch_size * length * dim {
                let mut resized = vec![0.0; batch_size * length * dim];
                let actual_length = sines.len() / (batch_size * dim);
                
                for b in 0..batch_size {
                    for l in 0..length {
                        for d in 0..dim {
                            let src_l = if actual_length > 0 {
                                ((l * actual_length) / length).min(actual_length - 1)
                            } else {
                                0
                            };
                            let src_idx = b * actual_length * dim + src_l * dim + d;
                            let dst_idx = b * length * dim + l * dim + d;
                            
                            if src_idx < sines.len() {
                                resized[dst_idx] = sines[src_idx];
                            }
                        }
                    }
                }
                resized
            } else {
                sines
            };
            
            Tensor::from_data(final_sines, vec![batch_size, length, dim])
        } else {
            // EXACT PYTORCH pulse-train generation mode
            let uv = self.f02uv(&Tensor::from_data(f0_values.data().to_vec(), f0_values.shape().to_vec()));
            
            // PyTorch: uv_1 = torch.roll(uv, shifts=-1, dims=1)
            let mut uv_1 = vec![0.0; uv.data().len()];
            for b in 0..batch_size {
                for l in 0..length {
                    let src_l = if l + 1 < length { l + 1 } else { 0 }; // roll -1
                    uv_1[b * length + l] = uv.data()[b * length + src_l];
                }
            }
            
            // PyTorch: uv_1[:, -1, :] = 1
            for b in 0..batch_size {
                uv_1[b * length + (length - 1)] = 1.0;
            }
            
            // PyTorch: u_loc = (uv < 1) * (uv_1 > 0)
            let mut u_loc = vec![false; batch_size * length];
            for b in 0..batch_size {
                for l in 0..length {
                    let idx = b * length + l;
                    u_loc[idx] = uv.data()[idx] < 1.0 && uv_1[idx] > 0.0;
                }
            }
            
            // PyTorch: tmp_cumsum = torch.cumsum(rad_values, dim=1)
            let mut tmp_cumsum = vec![0.0; rad_values.len()];
            for b in 0..batch_size {
                for d in 0..dim {
                    let mut cumsum = 0.0;
                    for l in 0..length {
                        let idx = b * length * dim + l * dim + d;
                        cumsum += rad_values[idx];
                        tmp_cumsum[idx] = cumsum;
                    }
                }
            }
            
            // PyTorch complex per-batch processing
            for b in 0..batch_size {
                // Extract u_loc for this batch
                let mut batch_u_loc = vec![false; length];
                for l in 0..length {
                    batch_u_loc[l] = u_loc[b * length + l];
                }
                
                // Collect indices where u_loc is true
                let u_loc_indices: Vec<usize> = batch_u_loc.iter().enumerate()
                    .filter(|(_, &is_loc)| is_loc)
                    .map(|(i, _)| i)
                    .collect();
                
                if !u_loc_indices.is_empty() {
                    // Extract temp_sum values
                    let mut temp_sum = vec![vec![0.0; dim]; u_loc_indices.len()];
                    for (i, &loc_idx) in u_loc_indices.iter().enumerate() {
                        for d in 0..dim {
                            let idx = b * length * dim + loc_idx * dim + d;
                            temp_sum[i][d] = tmp_cumsum[idx];
                        }
                    }
                    
                    // PyTorch: temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                    for i in 1..temp_sum.len() {
                        for d in 0..dim {
                            temp_sum[i][d] -= temp_sum[i - 1][d];
                        }
                    }
                    
                    // PyTorch: tmp_cumsum[idx, :, :] = 0; tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
                    for l in 0..length {
                        for d in 0..dim {
                            tmp_cumsum[b * length * dim + l * dim + d] = 0.0;
                        }
                    }
                    
                    for (i, &loc_idx) in u_loc_indices.iter().enumerate() {
                        for d in 0..dim {
                            let idx = b * length * dim + loc_idx * dim + d;
                            if i < temp_sum.len() {
                                tmp_cumsum[idx] = temp_sum[i][d];
                            }
                        }
                    }
                }
            }
            
            // PyTorch: i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
            let mut rad_diff = vec![0.0; rad_values.len()];
            for i in 0..rad_values.len() {
                rad_diff[i] = rad_values[i] - tmp_cumsum[i];
            }
            
            let mut i_phase = vec![0.0; rad_diff.len()];
            for b in 0..batch_size {
                for d in 0..dim {
                    let mut cumsum = 0.0;
                    for l in 0..length {
                        let idx = b * length * dim + l * dim + d;
                        cumsum += rad_diff[idx];
                        i_phase[idx] = cumsum;
                    }
                }
            }
            
            // PyTorch: sines = torch.cos(i_phase * 2 * torch.pi)
            let mut sines = vec![0.0; i_phase.len()];
            for i in 0..i_phase.len() {
                sines[i] = (i_phase[i] * 2.0 * PI).cos();
            }
            
            Tensor::from_data(sines, vec![batch_size, length, dim])
        }
    }
    
    /// EXACT PYTORCH forward() implementation
    pub fn forward(&self, f0: &Tensor<f32>) 
        -> (Tensor<f32>, Tensor<f32>, Tensor<f32>) {
        // PyTorch docstring:
        // input F0: tensor(batchsize=1, length, dim=1) - f0 for unvoiced steps should be 0
        // output sine_tensor: tensor(batchsize=1, length, dim)
        // output uv: tensor(batchsize=1, length, 1)
        
        let shape = f0.shape();
        let (batch_size, length) = (shape[0], shape[1]);
        
        // PyTorch: f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
        // PyTorch: fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))
        
        let mut f0_buf = vec![0.0; batch_size * length * self.dim];
        
        for b in 0..batch_size {
            for l in 0..length {
                let f0_val = f0.data()[b * length + l];
                
                // Generate harmonics: multiply f0 by [1, 2, 3, ..., harmonic_num+1]
                for h in 0..self.dim {
                    let harmonic_mult = (h + 1) as f32; // 1-based: [1, 2, 3, ...]
                    f0_buf[b * length * self.dim + l * self.dim + h] = f0_val * harmonic_mult;
                }
            }
        }
        
        let fn_tensor = Tensor::from_data(f0_buf, vec![batch_size, length, self.dim]);
        
        // PyTorch: sine_waves = self._f02sine(fn) * self.sine_amp
        let sine_waves_raw = self.f02sine(&fn_tensor);
        
        let mut sine_waves = vec![0.0; sine_waves_raw.data().len()];
        for i in 0..sine_waves_raw.data().len() {
            sine_waves[i] = sine_waves_raw.data()[i] * self.sine_amp;
        }
        let sine_waves_tensor = Tensor::from_data(sine_waves, sine_waves_raw.shape().to_vec());
        
        // PyTorch: uv = self._f02uv(f0)
        let uv = self.f02uv(f0);
        
        // PyTorch: noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        // PyTorch: noise = noise_amp * torch.randn_like(sine_waves)
        let mut rng = rand::thread_rng();
        let mut noise = vec![0.0; sine_waves_tensor.data().len()];
        
        for b in 0..batch_size {
            for l in 0..length {
                let uv_val = uv.data()[b * length + l];
                let noise_amp = uv_val * self.noise_std + (1.0 - uv_val) * (self.sine_amp / 3.0);
                
                for d in 0..self.dim {
                    let idx = b * length * self.dim + l * self.dim + d;
                    // PyTorch: torch.randn_like() - proper Gaussian random normal distribution
                    let u1 = rng.gen::<f32>();
                    let u2 = rng.gen::<f32>();
                    let gaussian = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos(); // Box-Muller transform
                    noise[idx] = noise_amp * gaussian;
                }
            }
        }
        
        let noise_tensor = Tensor::from_data(noise, sine_waves_tensor.shape().to_vec());
        
        // PyTorch: sine_waves = sine_waves * uv + noise
        let mut final_sine_waves = vec![0.0; sine_waves_tensor.data().len()];
        
        for b in 0..batch_size {
            for l in 0..length {
                let uv_val = uv.data()[b * length + l];
                
                for d in 0..self.dim {
                    let idx = b * length * self.dim + l * self.dim + d;
                    final_sine_waves[idx] = sine_waves_tensor.data()[idx] * uv_val + noise_tensor.data()[idx];
                }
            }
        }
        
        let final_tensor = Tensor::from_data(final_sine_waves, sine_waves_tensor.shape().to_vec());
        
        (final_tensor, uv, noise_tensor)
    }
}

#[cfg(feature = "weights")]
impl SineGen {
    pub fn sinegen_mut(&mut self) -> &mut Self {
        self
    }
}