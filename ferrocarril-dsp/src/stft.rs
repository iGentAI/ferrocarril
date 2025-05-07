//! Custom STFT/iSTFT implementation using conv1d operations

use ferrocarril_core::tensor::Tensor;
use std::f32::consts::PI;

pub struct CustomSTFT {
    filter_length: usize,
    hop_length: usize,
    win_length: usize,
    window: Vec<f32>,
    freq_bins: usize,
    // Pre-computed DFT matrices
    weight_forward_real: Tensor<f32>,
    weight_forward_imag: Tensor<f32>,
    weight_backward_real: Tensor<f32>,
    weight_backward_imag: Tensor<f32>,
}

impl CustomSTFT {
    pub fn new(filter_length: usize, hop_length: usize, win_length: usize) -> Self {
        let window = hann_window(win_length, true);
        let freq_bins = filter_length / 2 + 1;
        
        // Precompute forward DFT matrices
        let (forward_real, forward_imag) = Self::compute_dft_matrices(filter_length, freq_bins, &window);
        
        // Precompute backward DFT matrices
        let (backward_real, backward_imag) = Self::compute_idft_matrices(filter_length, freq_bins, &window);
        
        Self {
            filter_length,
            hop_length,
            win_length,
            window,
            freq_bins,
            weight_forward_real: forward_real,
            weight_forward_imag: forward_imag,
            weight_backward_real: backward_real,
            weight_backward_imag: backward_imag,
        }
    }
    
    fn compute_dft_matrices(n_fft: usize, freq_bins: usize, window: &[f32]) -> (Tensor<f32>, Tensor<f32>) {
        let mut real_matrix = vec![0.0; freq_bins * n_fft];
        let mut imag_matrix = vec![0.0; freq_bins * n_fft];
        
        // Compute DFT matrix
        for k in 0..freq_bins {
            for n in 0..n_fft {
                let angle = 2.0 * PI * k as f32 * n as f32 / n_fft as f32;
                real_matrix[k * n_fft + n] = angle.cos() * window[n];
                imag_matrix[k * n_fft + n] = -angle.sin() * window[n];
            }
        }
        
        (
            Tensor::from_data(real_matrix, vec![freq_bins, 1, n_fft]),
            Tensor::from_data(imag_matrix, vec![freq_bins, 1, n_fft]),
        )
    }
    
    fn compute_idft_matrices(n_fft: usize, freq_bins: usize, window: &[f32]) -> (Tensor<f32>, Tensor<f32>) {
        let inv_scale = 1.0 / n_fft as f32;
        let mut real_matrix = vec![0.0; freq_bins * n_fft];
        let mut imag_matrix = vec![0.0; freq_bins * n_fft];
        
        // Compute iDFT matrix
        for n in 0..n_fft {
            for k in 0..freq_bins {
                let angle = 2.0 * PI * n as f32 * k as f32 / n_fft as f32;
                real_matrix[k * n_fft + n] = angle.cos() * window[n] * inv_scale;
                imag_matrix[k * n_fft + n] = angle.sin() * window[n] * inv_scale;
            }
        }
        
        (
            Tensor::from_data(real_matrix, vec![freq_bins, 1, n_fft]),
            Tensor::from_data(imag_matrix, vec![freq_bins, 1, n_fft]),
        )
    }
    
    pub fn transform(&self, waveform: &Tensor<f32>) -> (Tensor<f32>, Tensor<f32>) {
        // Check input shape and ensure it's valid
        let shape = waveform.shape();
        
        println!("STFT input shape: {:?}", shape);
        
        let (batch_size, signal_length) = match shape.len() {
            1 => (1, shape[0]),                  // (T)
            2 => (shape[0], shape[1]),           // (B, T)
            3 if shape[1] == 1 => {
                // Special case for [B, 1, T] which should be treated as [B, T]
                println!("Handling 3D tensor with shape {:?} as [B, T]", shape);
                (shape[0], shape[2])
            },
            _ => {
                println!("Invalid waveform shape: {:?}", shape);
                panic!("waveform must have shape (T) or (B, T) or (B, 1, T), but got {:?}", shape)
            }
        };
        
        // Pad signal
        let pad_len = self.filter_length / 2;
        let padded_length = signal_length + 2 * pad_len;
        
        // Create padded tensor
        let mut padded_data = vec![0.0; batch_size * padded_length];
        
        // Copy signal with padding
        for b in 0..batch_size {
            // Copy the actual signal
            for i in 0..signal_length {
                let idx = match shape.len() {
                    1 => &[i],                  // 1D tensor: direct index
                    2 => &[b, i],               // 2D tensor: [b, i]
                    _ => &[b, 0, i]             // 3D tensor with singleton dimension: [b, 0, i]
                };
                
                padded_data[b * padded_length + pad_len + i] = waveform[idx];
            }
            // Simple reflection padding
            for i in 0..pad_len {
                let left_idx = match shape.len() {
                    1 => &[pad_len - i - 1],
                    2 => &[b, pad_len - i - 1],
                    _ => &[b, 0, pad_len - i - 1]
                };
                
                let right_idx = match shape.len() {
                    1 => &[signal_length - i - 1],
                    2 => &[b, signal_length - i - 1],
                    _ => &[b, 0, signal_length - i - 1]
                };
                
                padded_data[b * padded_length + i] = waveform[left_idx];
                padded_data[b * padded_length + padded_length - i - 1] = waveform[right_idx];
            }
        }
        
        let padded_signal = Tensor::from_data(padded_data, vec![batch_size, 1, padded_length]);
        
        // Perform convolution using conv1d-style computation
        let num_frames = (padded_length - self.filter_length) / self.hop_length + 1;
        
        let mut real_output = vec![0.0; batch_size * self.freq_bins * num_frames];
        let mut imag_output = vec![0.0; batch_size * self.freq_bins * num_frames];
        
        // Convolve for each frame
        for b in 0..batch_size {
            for frame in 0..num_frames {
                let start_idx = frame * self.hop_length;
                
                // Convolve with each frequency bin's kernel
                for k in 0..self.freq_bins {
                    let mut real_sum = 0.0;
                    let mut imag_sum = 0.0;
                    
                    for n in 0..self.filter_length {
                        let signal_idx = b * padded_length + start_idx + n;
                        let kernel_idx = k * self.filter_length + n;
                        
                        real_sum += padded_data[signal_idx] * self.weight_forward_real.data()[kernel_idx];
                        imag_sum += padded_data[signal_idx] * self.weight_forward_imag.data()[kernel_idx];
                    }
                    
                    real_output[b * self.freq_bins * num_frames + k * num_frames + frame] = real_sum;
                    imag_output[b * self.freq_bins * num_frames + k * num_frames + frame] = imag_sum;
                }
            }
        }
        
        // Compute magnitude and phase
        let mut magnitude = vec![0.0; batch_size * self.freq_bins * num_frames];
        let mut phase = vec![0.0; batch_size * self.freq_bins * num_frames];
        
        for i in 0..magnitude.len() {
            let real = real_output[i];
            let imag = imag_output[i];
            
            magnitude[i] = (real * real + imag * imag + 1e-14).sqrt();
            phase[i] = imag.atan2(real);
            
            // Handle edge case for ONNX compatibility
            if imag == 0.0 && real < 0.0 {
                phase[i] = PI;
            }
        }
        
        (
            Tensor::from_data(magnitude, vec![batch_size, self.freq_bins, num_frames]),
            Tensor::from_data(phase, vec![batch_size, self.freq_bins, num_frames]),
        )
    }
    
    pub fn inverse(&self, magnitude: &Tensor<f32>, phase: &Tensor<f32>) -> Tensor<f32> {
        let shape = magnitude.shape();
        let (batch_size, freq_bins, num_frames) = (shape[0], shape[1], shape[2]);
        
        // Recreate real and imaginary parts
        let mut real_part = vec![0.0; batch_size * freq_bins * num_frames];
        let mut imag_part = vec![0.0; batch_size * freq_bins * num_frames];
        
        for b in 0..batch_size {
            for k in 0..freq_bins {
                for f in 0..num_frames {
                    let idx = b * freq_bins * num_frames + k * num_frames + f;
                    let mag = magnitude[&[b, k, f]];
                    let ph = phase[&[b, k, f]];
                    
                    real_part[idx] = mag * ph.cos();
                    imag_part[idx] = mag * ph.sin();
                }
            }
        }
        
        // Perform inverse STFT
        let output_length = (num_frames - 1) * self.hop_length + self.filter_length;
        let mut output = vec![0.0; batch_size * output_length];
        
        // Apply inverse DFT
        for b in 0..batch_size {
            for frame in 0..num_frames {
                let out_start = frame * self.hop_length;
                
                for n in 0..self.filter_length {
                    let out_idx = b * output_length + out_start + n;
                    
                    // Sum over frequency bins
                    let mut sum = 0.0;
                    for k in 0..freq_bins {
                        let spec_idx = b * freq_bins * num_frames + k * num_frames + frame;
                        let kernel_idx = k * self.filter_length + n;
                        
                        sum += real_part[spec_idx] * self.weight_backward_real.data()[kernel_idx]
                            - imag_part[spec_idx] * self.weight_backward_imag.data()[kernel_idx];
                    }
                    
                    output[out_idx] += sum;
                }
            }
        }
        
        // Remove padding
        let pad_len = self.filter_length / 2;
        let unpadded_length = output_length - 2 * pad_len;
        let mut unpadded_output = vec![0.0; batch_size * unpadded_length];
        
        for b in 0..batch_size {
            for i in 0..unpadded_length {
                unpadded_output[b * unpadded_length + i] = output[b * output_length + pad_len + i];
            }
        }
        
        Tensor::from_data(unpadded_output, vec![batch_size, unpadded_length])
    }
}

/// Create a Hann window
pub fn hann_window(size: usize, periodic: bool) -> Vec<f32> {
    let mut window = vec![0.0; size];
    let n = if periodic { size } else { size - 1 };
    
    for i in 0..size {
        window[i] = 0.5 * (1.0 - (2.0 * PI * i as f32 / n as f32).cos());
    }
    
    window
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stft_transform() {
        let stft = CustomSTFT::new(16, 4, 16);
        let waveform = Tensor::from_data(vec![0.0; 128], vec![128]);
        let (mag, phase) = stft.transform(&waveform);
        
        assert_eq!(mag.shape()[0], 1);
        assert_eq!(mag.shape()[1], 9); // filter_length/2 + 1
    }
}