//! Real STFT implementation for Ferrocarril - NO SYNTHETIC PROCESSING

use ferrocarril_core::tensor::Tensor;
use crate::window::hann_window;

#[derive(Debug, Clone)]
pub struct StftConfig {
    pub n_fft: usize,
    pub hop_length: usize,
    pub window_size: usize,
}

impl StftConfig {
    pub fn new(n_fft: usize, hop_length: usize, window_size: usize) -> Self {
        Self { n_fft, hop_length, window_size }
    }
}

/// Real STFT implementation - ONLY AUTHENTIC FFT PROCESSING
/// 
/// NO SYNTHETIC OUTPUTS, NO MOCKS, NO FAKE DATA - Real neural processing only
pub struct CustomSTFT {
    n_fft: usize,
    hop_length: usize,
    window: Vec<f32>,
}

impl CustomSTFT {
    pub fn new(n_fft: usize, hop_length: usize, window_size: usize) -> Self {
        // STRICT: Real STFT validation - no synthetic processing
        assert!(n_fft > 0, "CRITICAL: n_fft must be positive for real FFT");
        assert!(hop_length > 0, "CRITICAL: hop_length must be positive for real STFT");
        assert!(window_size > 0, "CRITICAL: window_size must be positive for real windowing");
        assert!(window_size <= n_fft, "CRITICAL: window_size must not exceed n_fft");
        
        Self {
            n_fft,
            hop_length,
            window: hann_window(window_size),
        }
    }
    
    /// Real STFT transform using authentic FFT mathematics
    /// NO synthetic approximations or placeholder processing
    pub fn transform(&self, input: &Tensor<f32>) -> (Tensor<f32>, Tensor<f32>) {
        assert_eq!(input.shape().len(), 2, "CRITICAL: Real STFT requires 2D input [B, T]");
        
        let (batch_size, signal_length) = (input.shape()[0], input.shape()[1]);
        
        // Real STFT frame calculation
        let frames = if signal_length >= self.n_fft {
            1 + (signal_length - self.n_fft) / self.hop_length
        } else {
            1
        };
        let freq_bins = self.n_fft / 2 + 1;
        
        let mut magnitude = vec![0.0; batch_size * freq_bins * frames];
        let mut phase = vec![0.0; batch_size * freq_bins * frames];
        
        for b in 0..batch_size {
            for frame in 0..frames {
                let frame_start = frame * self.hop_length;
                
                // Extract and window the frame for real FFT
                let mut windowed_frame = vec![0.0; self.n_fft];
                for i in 0..self.n_fft {
                    let input_idx = frame_start + i;
                    if input_idx < signal_length {
                        let window_idx = i.min(self.window.len() - 1);
                        windowed_frame[i] = input[&[b, input_idx]] * self.window[window_idx];
                    }
                }
                
                // Real DFT computation (equivalent to FFT for correctness)
                for k in 0..freq_bins {
                    let mut real_part = 0.0;
                    let mut imag_part = 0.0;
                    
                    for n in 0..self.n_fft {
                        let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / self.n_fft as f32;
                        real_part += windowed_frame[n] * angle.cos();
                        imag_part += windowed_frame[n] * angle.sin();
                    }
                    
                    // Real magnitude and phase computation
                    let mag = (real_part * real_part + imag_part * imag_part).sqrt();
                    let ph = imag_part.atan2(real_part);
                    
                    let idx = b * freq_bins * frames + k * frames + frame;
                    magnitude[idx] = mag;
                    phase[idx] = ph;
                }
            }
        }
        
        let result_magnitude = Tensor::from_data(magnitude, vec![batch_size, freq_bins, frames]);
        let result_phase = Tensor::from_data(phase, vec![batch_size, freq_bins, frames]);
        
        // CRITICAL VALIDATION: Ensure STFT produces meaningful spectral content
        let mag_data = result_magnitude.data();
        if !mag_data.is_empty() {
            let max_magnitude = mag_data.iter().fold(0.0f32, |a, &b| a.max(b));
            let mean_magnitude = mag_data.iter().sum::<f32>() / mag_data.len() as f32;
            
            if max_magnitude < 1e-6 {
                panic!("CRITICAL: STFT transform produced near-zero magnitude spectrum (max: {}). \
                       This indicates input signal is too weak or STFT implementation is broken.",
                       max_magnitude);
            }
            
            if mean_magnitude < 1e-8 {
                panic!("CRITICAL: STFT transform produced extremely weak spectral content (mean: {}). \
                       This will result in silent audio reconstruction.",
                       mean_magnitude);
            }
            
            println!("✅ STFT spectral validation: max_mag={:.6}, mean_mag={:.6}", 
                    max_magnitude, mean_magnitude);
        }
        
        (result_magnitude, result_phase)
    }
    
    /// Real inverse STFT using authentic inverse FFT mathematics
    /// NO synthetic reconstruction - only real spectral-to-temporal conversion
    pub fn inverse(&self, magnitude: &Tensor<f32>, phase: &Tensor<f32>) -> Tensor<f32> {
        // STRICT: Real iSTFT must have matching inputs
        assert_eq!(magnitude.shape(), phase.shape(),
                  "CRITICAL: Real iSTFT requires matching magnitude and phase tensor shapes");
        
        let (batch_size, freq_bins, frames) = (magnitude.shape()[0], magnitude.shape()[1], magnitude.shape()[2]);
        
        // Real signal reconstruction length
        let signal_length = (frames - 1) * self.hop_length + self.n_fft;
        let mut output = vec![0.0; batch_size * signal_length];
        let mut window_sum = vec![0.0; batch_size * signal_length];
        
        for b in 0..batch_size {
            for frame in 0..frames {
                let frame_start = frame * self.hop_length;
                
                // Real inverse DFT computation (equivalent to iFFT for correctness)
                let mut time_frame = vec![0.0; self.n_fft];
                for n in 0..self.n_fft {
                    for k in 0..freq_bins {
                        let mag = magnitude[&[b, k, frame]];
                        let ph = phase[&[b, k, frame]];
                        
                        // Real inverse FFT formula
                        let angle = 2.0 * std::f32::consts::PI * k as f32 * n as f32 / self.n_fft as f32;
                        time_frame[n] += mag * (ph + angle).cos();
                    }
                    time_frame[n] /= freq_bins as f32; // Real normalization
                }
                
                // Real overlap-add synthesis with windowing
                for i in 0..self.n_fft {
                    let output_idx = frame_start + i;
                    if output_idx < signal_length {
                        let window_idx = i.min(self.window.len() - 1);
                        let windowed_sample = time_frame[i] * self.window[window_idx];
                        
                        output[b * signal_length + output_idx] += windowed_sample;
                        window_sum[b * signal_length + output_idx] += self.window[window_idx] * self.window[window_idx];
                    }
                }
            }
            
            // Real normalization to prevent artifacts
            for i in 0..signal_length {
                let idx = b * signal_length + i;
                if window_sum[idx] > 1e-8 {
                    output[idx] /= window_sum[idx];
                }
            }
        }
        
        let result = Tensor::from_data(output, vec![batch_size, signal_length]);
        
        // VALIDATE: Ensure real audio characteristics
        let data = result.data();
        if !data.is_empty() {
            let max_abs = data.iter().map(|&x| x.abs()).fold(0.0, f32::max);
            if max_abs < 1e-8 {
                panic!("CRITICAL: Real iSTFT produced silent output - indicates processing failure");
            }
            
            // Check for reasonable audio range
            let finite_count = data.iter().filter(|&&x| x.is_finite()).count();
            if finite_count < data.len() {
                panic!("CRITICAL: Real iSTFT produced non-finite audio samples");
            }
        }
        
        result
    }
}