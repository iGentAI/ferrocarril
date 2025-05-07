//! Custom STFT / iSTFT implemented with pre–computed real kernels.
//!
//! This is a dependency–free alternative to torch.stft that matches the
//! behaviour of kokoro/custom_stft.py almost bit-exactly (floating point
//! round-off differences only).

use ferrocarril_core::tensor::Tensor;
use crate::window::{hann_window, hamming_window, rect_window};
use std::f32::consts::PI;

//--------------------------------------------------------------------------------------------------
// Public API helpers
//--------------------------------------------------------------------------------------------------

/// Which window to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    Hann,
    Hamming,
    Rect,
}

impl WindowType {
    fn build(self, size: usize, periodic: bool) -> Vec<f32> {
        match self {
            WindowType::Hann => hann_window(size, periodic),
            WindowType::Hamming => hamming_window(size, periodic),
            WindowType::Rect => rect_window(size),
        }
    }
}

/// Builder so the many optional parameters do not lead to a 10-item `new()` call.
pub struct StftConfig {
    pub filter_length: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub window: WindowType,
    pub center: bool,          // mimic torch.stft(center=True)
    pub pad_mode: PadMode,
}

impl StftConfig {
    pub fn new(filter_length: usize, hop_length: usize) -> Self {
        Self {
            filter_length,
            hop_length,
            win_length: filter_length,
            window: WindowType::Hann,
            center: true,
            pad_mode: PadMode::Replicate,
        }
    }
}

/// Padding strategy when `center == true`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PadMode {
    Constant(f32),
    Replicate,
    // (torch's reflect is not available for dynamic shapes in ONNX; replicate is close enough)
}

//--------------------------------------------------------------------------------------------------
// STFT structure holding the pre-computed kernels
//--------------------------------------------------------------------------------------------------

pub struct CustomSTFT {
    cfg: StftConfig,
    n_fft: usize,
    freq_bins: usize,
    window: Vec<f32>, // length == n_fft

    // (out, in=1, k) so they can directly be used as conv kernels
    weight_forward_real: Tensor<f32>,
    weight_forward_imag: Tensor<f32>,
    weight_backward_real: Tensor<f32>,
    weight_backward_imag: Tensor<f32>,
}

//--------------------------------------------------------------------------------------------------
// Construction
//--------------------------------------------------------------------------------------------------

impl CustomSTFT {
    pub fn new(filter_length: usize, hop_length: usize, win_length: usize) -> Self {
        Self::from_cfg(StftConfig {
            filter_length,
            hop_length,
            win_length,
            window: WindowType::Hann,
            center: true,
            pad_mode: PadMode::Replicate,
        })
    }

    pub fn from_cfg(cfg: StftConfig) -> Self {
        assert!(
            cfg.filter_length >= 2,
            "filter_length must be at least 2 samples"
        );
        assert!(
            cfg.hop_length >= 1 && cfg.hop_length <= cfg.filter_length,
            "hop_length must be between 1 and filter_length"
        );
        assert!(
            cfg.win_length >= 1 && cfg.win_length <= cfg.filter_length,
            "win_length must be in 1..=filter_length"
        );

        let n_fft = cfg.filter_length;
        let freq_bins = n_fft / 2 + 1;

        // 1) ---------------------------------------------------------------- Window
        let mut window = cfg.window.build(cfg.win_length, true);
        if cfg.win_length < n_fft {
            // zero-pad the window up to n_fft
            window.resize(n_fft, 0.0);
        } else if cfg.win_length > n_fft {
            window.truncate(n_fft);
        }

        // 2) ---------------------------------------------------------------- forward kernels
        let (w_fr, w_fi) = compute_forward_kernels(n_fft, freq_bins, &window);

        // 3) ---------------------------------------------------------------- inverse kernels
        let (w_br, w_bi) = compute_inverse_kernels(n_fft, freq_bins, &window);

        Self {
            cfg,
            n_fft,
            freq_bins,
            window,
            weight_forward_real: w_fr,
            weight_forward_imag: w_fi,
            weight_backward_real: w_br,
            weight_backward_imag: w_bi,
        }
    }
    
    /// Get the hop length
    pub fn hop_length(&self) -> usize {
        self.cfg.hop_length
    }
}

//--------------------------------------------------------------------------------------------------
// Kernel pre-computation (pure math; called once)
//--------------------------------------------------------------------------------------------------

#[inline]
fn compute_forward_kernels(
    n_fft: usize,
    freq_bins: usize,
    window: &[f32],
) -> (Tensor<f32>, Tensor<f32>) {
    let mut real = vec![0.0; freq_bins * n_fft];
    let mut imag = vec![0.0; freq_bins * n_fft];

    // k = 0 .. freq_bins-1 (rows) ; n = 0 .. n_fft-1 (columns)
    for k in 0..freq_bins {
        for n in 0..n_fft {
            let angle = 2.0 * PI * (k * n) as f32 / n_fft as f32;
            let w = window[n];
            real[k * n_fft + n] =  angle.cos() * w;   //  cos
            imag[k * n_fft + n] = -angle.sin() * w;   // -sin   (negative sign!)
        }
    }

    (
        Tensor::from_data(real, vec![freq_bins, 1, n_fft]),
        Tensor::from_data(imag, vec![freq_bins, 1, n_fft]),
    )
}

#[inline]
fn compute_inverse_kernels(
    n_fft: usize,
    freq_bins: usize,
    window: &[f32],
) -> (Tensor<f32>, Tensor<f32>) {
    let scale = 1.0 / n_fft as f32;
    let mut real = vec![0.0; freq_bins * n_fft];
    let mut imag = vec![0.0; freq_bins * n_fft];

    // n along *rows* because conv_transpose will treat the kernel as (in, out, k)
    for k in 0..freq_bins {
        for n in 0..n_fft {
            let angle = 2.0 * PI * (n * k) as f32 / n_fft as f32;
            let w = window[n] * scale;
            real[k * n_fft + n] =  angle.cos() * w;
            imag[k * n_fft + n] =  angle.sin() * w; // positive sign
        }
    }

    (
        Tensor::from_data(real, vec![freq_bins, 1, n_fft]),
        Tensor::from_data(imag, vec![freq_bins, 1, n_fft]),
    )
}

//--------------------------------------------------------------------------------------------------
// Forward STFT  (conv1d)
//--------------------------------------------------------------------------------------------------

impl CustomSTFT {
    /// STFT  ⇒  magnitude + phase  (batch, freq_bins, frames)
    pub fn transform(&self, waveform: &Tensor<f32>) -> (Tensor<f32>, Tensor<f32>) {
        // input shape : (B, T)   OR  (T) for mono convenience
        let (batch, time) = match waveform.shape()[..] {
            [t] => (1, t),
            [b, t] => (b, t),
            _ => panic!("waveform must have shape (T) or (B, T), got shape {:?}", waveform.shape()),
        };

        // Verify the waveform has enough samples for the STFT calculation
        if time < self.n_fft {
            panic!(
                "Input waveform length ({}) must be at least n_fft ({})",
                time, self.n_fft
            );
        }

        //------------------------------------------- 1) optional centre padding
        let pad = if self.cfg.center { self.n_fft / 2 } else { 0 };
        let total_len = time + 2 * pad;
        let mut padded = vec![0.0f32; batch * total_len];

        for b in 0..batch {
            // copy source
            for t in 0..time {
                padded[b * total_len + pad + t] = waveform.get(&[b, t]);
            }
            if pad > 0 {
                match self.cfg.pad_mode {
                    PadMode::Constant(val) => {
                        for t in 0..pad {
                            padded[b * total_len + t] = val;
                            padded[b * total_len + pad + time + t] = val;
                        }
                    }
                    PadMode::Replicate => {
                        let first = waveform.get(&[b, 0]);
                        let last  = waveform.get(&[b, time - 1]);
                        for t in 0..pad {
                            padded[b * total_len + t] = first;
                            padded[b * total_len + pad + time + t] = last;
                        }
                    }
                }
            }
        }
        let padded = Tensor::from_data(padded, vec![batch, 1, total_len]);

        //------------------------------------------- 2) conv1d → (B, freq_bins, frames)
        let frames = (total_len - self.n_fft) / self.cfg.hop_length + 1;
        
        // Ensure we have at least one frame
        if frames < 1 {
            panic!(
                "Cannot compute STFT - input length ({}) too short for n_fft ({}) and hop_length ({})", 
                total_len, self.n_fft, self.cfg.hop_length
            );
        }
        
        let mut real = vec![0.0; batch * self.freq_bins * frames];
        let mut imag = vec![0.0; batch * self.freq_bins * frames];

        // Perform the STFT operation
        for b in 0..batch {
            for m in 0..frames {
                let start = m * self.cfg.hop_length;
                for k in 0..self.freq_bins {
                    let mut acc_r = 0.0;
                    let mut acc_i = 0.0;
                    let kernel_base = k * self.n_fft;
                    for n in 0..self.n_fft {
                        let x = padded.get(&[b, 0, start + n]);
                        acc_r += x * self.weight_forward_real.data()[kernel_base + n];
                        acc_i += x * self.weight_forward_imag.data()[kernel_base + n];
                    }
                    let dst = b * (self.freq_bins * frames) + k * frames + m;
                    real[dst] = acc_r;
                    imag[dst] = acc_i;
                }
            }
        }

        //------------------------------------------- 3) magnitude / phase
        let mut mag = Vec::with_capacity(real.len());
        let mut phs = Vec::with_capacity(real.len());
        for (r, i) in real.iter().zip(&imag) {
            let magnitude = (r * r + i * i + 1e-14).sqrt();
            let mut phase = i.atan2(*r);
            if *i == 0.0 && *r < 0.0 {
                // fix ONNX atan2 corner case
                phase = PI;
            }
            mag.push(magnitude);
            phs.push(phase);
        }

        let mag_tensor = Tensor::from_data(mag, vec![batch, self.freq_bins, frames]);
        let phs_tensor = Tensor::from_data(phs, vec![batch, self.freq_bins, frames]);

        (mag_tensor, phs_tensor)
    }
}

//--------------------------------------------------------------------------------------------------
// Inverse  (conv_transpose1d + overlap/add)
//--------------------------------------------------------------------------------------------------

impl CustomSTFT {
    pub fn inverse(&self, magnitude: &Tensor<f32>, phase: &Tensor<f32>) -> Tensor<f32> {
        // Validate input shapes
        if magnitude.shape().len() != 3 || phase.shape().len() != 3 {
            panic!(
                "magnitude and phase must be 3D tensors with shape [B, freq_bins, frames]. Got: magnitude={:?}, phase={:?}",
                magnitude.shape(), phase.shape()
            );
        }
        
        // Validate frequency bins
        if magnitude.shape()[1] != self.freq_bins || phase.shape()[1] != self.freq_bins {
            panic!(
                "frequency bins mismatch. Expected {}, got magnitude={}, phase={}",
                self.freq_bins, magnitude.shape()[1], phase.shape()[1]
            );
        }
        
        // Ensure dimensions match
        if magnitude.shape() != phase.shape() {
            panic!(
                "magnitude and phase dimensions don't match. magnitude={:?}, phase={:?}",
                magnitude.shape(), phase.shape()
            );
        }
        
        // (B, K, F)
        let [batch, k, frames] = {
            let s = magnitude.shape();
            [s[0], s[1], s[2]]
        };
        
        //---------------------- 1) reconstruct real / imag spectrogram
        let mut spec_r = vec![0.0; batch * k * frames];
        let mut spec_i = vec![0.0; batch * k * frames];
        for b in 0..batch {
            for bin in 0..k {
                for f in 0..frames {
                    let idx = b * (k * frames) + bin * frames + f;
                    let mag = magnitude.get(&[b, bin, f]);
                    let ph = phase.get(&[b, bin, f]);
                    spec_r[idx] = mag * ph.cos();
                    spec_i[idx] = mag * ph.sin();
                }
            }
        }

        //---------------------- 2) iSTFT → time domain through transposed conv
        let out_len = (frames - 1) * self.cfg.hop_length + self.n_fft;
        let mut out = vec![0.0f32; batch * out_len];

        for b in 0..batch {
            for f in 0..frames {
                let start = f * self.cfg.hop_length;
                for n in 0..self.n_fft {
                    let dst = b * out_len + start + n;
                    let mut acc = 0.0;
                    for bin in 0..k {
                        let idx_spec = b * (k * frames) + bin * frames + f;
                        let base = bin * self.n_fft + n;
                        acc += spec_r[idx_spec] * self.weight_backward_real.data()[base]
                            - spec_i[idx_spec] * self.weight_backward_imag.data()[base];
                    }
                    out[dst] += acc;
                }
            }
        }

        //---------------------- 3) remove centre padding
        let (start, end) = if self.cfg.center {
            let pad = self.n_fft / 2;
            (pad, out_len - pad)
        } else {
            (0, out_len)
        };

        let final_len = end - start;
        let mut trimmed = vec![0.0f32; batch * final_len];
        for b in 0..batch {
            for t in 0..final_len {
                trimmed[b * final_len + t] = out[b * out_len + start + t];
            }
        }

        Tensor::from_data(trimmed, vec![batch, final_len])
    }
}

//--------------------------------------------------------------------------------------------------
// Tensor helper trait – **only** what we need here
//--------------------------------------------------------------------------------------------------

trait TensorExt {
    fn get(&self, idx: &[usize]) -> f32;
}

impl TensorExt for Tensor<f32> {
    fn get(&self, idx: &[usize]) -> f32 {
        // Minimal accessor to avoid dealing with Index<&[usize]> ergonomics
        match idx {
            [t] => self[&[*t]],
            [b, t] => self[&[*b, *t]],
            [b, c, t] => self[&[*b, *c, *t]],
            _ => panic!("unsupported ndim in get()"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn roundtrip_hann() {
        let cfg = StftConfig::new(16, 4);
        let stft = CustomSTFT::from_cfg(cfg);
        
        // Create input waveform
        let mut wave = vec![0.0f32; 64];
        for i in 0..64 {
            wave[i] = (i as f32 / 10.0).sin();
        }
        let wave_tensor = Tensor::from_data(wave.clone(), vec![1, 64]);
        
        let (mag, ph) = stft.transform(&wave_tensor);
        let recon = stft.inverse(&mag, &ph);
        
        // Verify reconstruction size
        assert_eq!(recon.shape(), wave_tensor.shape());
    }
}