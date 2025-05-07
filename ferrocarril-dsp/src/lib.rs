//! Digital Signal Processing components for Ferrocarril

pub mod stft;
pub mod window;

use ferrocarril_core::tensor::Tensor;

/// Sample rate used for audio processing
pub const SAMPLE_RATE: usize = 24000;

/// Convert a tensor of audio samples to a WAV file
pub fn save_wav(audio: &Tensor<f32>, path: &str) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;
    
    let data = audio.data();
    let num_samples = data.len();
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let byte_rate = SAMPLE_RATE as u32 * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align: u16 = num_channels * bits_per_sample / 8;
    
    let mut file = File::create(path)?;
    
    // Write WAV header
    file.write_all(b"RIFF")?;
    file.write_all(&((36 + num_samples * 2) as u32).to_le_bytes())?; // File size - 8
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // Subchunk1Size
    file.write_all(&1u16.to_le_bytes())?; // AudioFormat (PCM = 1)
    file.write_all(&num_channels.to_le_bytes())?;
    file.write_all(&(SAMPLE_RATE as u32).to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&bits_per_sample.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&(num_samples as u32 * 2).to_le_bytes())?; // Subchunk2Size
    
    // Write audio data
    for &sample in data {
        // Convert f32 to i16
        let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        file.write_all(&sample_i16.to_le_bytes())?;
    }
    
    Ok(())
}