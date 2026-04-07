//! SourceModuleHnNSF - Neural Source-Filter Model

use crate::{Forward, linear::Linear};
use ferrocarril_core::tensor::Tensor;
use super::sinegen::SineGen;
use rand::Rng;

/// SourceModuleHnNSF for generating source excitation 
/// based on F0 input
pub struct SourceModuleHnNSF {
    sinegen: SineGen,
    linear: Linear,
    tanh_fn: fn(f32) -> f32,
    harmonic_num: usize,
    sine_amp: f32,
}

impl SourceModuleHnNSF {
    pub fn new(
        sampling_rate: usize,
        upsample_scale: usize,
        harmonic_num: usize,
        sine_amp: f32,
        add_noise_std: f32,
        voiced_threshold: f32
    ) -> Self {
        let sinegen = SineGen::new(
            sampling_rate,
            upsample_scale,
            harmonic_num,
            sine_amp,
            add_noise_std,
            voiced_threshold,
            false // flag_for_pulse
        );
        
        // Linear layer to merge source harmonics
        let linear = Linear::new(harmonic_num + 1, 1, false);
        
        Self {
            sinegen,
            linear,
            tanh_fn: f32::tanh,
            harmonic_num,
            sine_amp,
        }
    }
    
    #[cfg(feature = "weights")]
    pub fn linear_mut(&mut self) -> &mut Linear {
        &mut self.linear
    }
    
    pub fn forward(&self, f0: &Tensor<f32>)
        -> (Tensor<f32>, Tensor<f32>, Tensor<f32>) {
        // Generate sine waves and noise
        let (sine_waves, uv, _) = self.sinegen.forward(f0);
        
        // Reshape sine_waves for the linear layer
        let (batch, time, _) = {
            let shape = sine_waves.shape();
            (shape[0], shape[1], shape[2])
        };
        
        let mut sine_merged = vec![0.0; batch * time];
        
        // Process each batch and time step
        for b in 0..batch {
            for t in 0..time {
                // Extract harmonics for this time step
                let mut harmonics = vec![0.0; self.harmonic_num + 1];
                for h in 0..self.harmonic_num + 1 {
                    harmonics[h] = sine_waves[&[b, t, h]];
                }
                
                // Pass through linear layer
                let harmonic_tensor = Tensor::from_data(harmonics.clone(), vec![1, self.harmonic_num + 1]);
                let merged = self.linear.forward(&harmonic_tensor);
                
                // Apply tanh activation
                sine_merged[b * time + t] = (self.tanh_fn)(merged[&[0, 0]]);
            }
        }
        
        // Generate noise source
        let mut rng = rand::thread_rng();
        let mut noise_source = vec![0.0; batch * time];
        for i in 0..batch * time {
            noise_source[i] = (rng.gen::<f32>() * 2.0 - 1.0) * self.sine_amp / 3.0;
        }
        
        (
            Tensor::from_data(sine_merged, vec![batch, time]),
            Tensor::from_data(noise_source, vec![batch, time]),
            uv
        )
    }
}