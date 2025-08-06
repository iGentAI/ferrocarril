//! Window functions for signal processing

/// Generate a Hann window of the specified size
pub fn hann_window(size: usize) -> Vec<f32> {
    if size == 0 {
        return vec![];
    }
    if size == 1 {
        return vec![1.0];
    }
    
    (0..size)
        .map(|i| {
            let phase = 2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32;
            0.5 * (1.0 - phase.cos())
        })
        .collect()
}

/// Generate a Hamming window of the specified size
pub fn hamming_window(size: usize) -> Vec<f32> {
    if size == 0 {
        return vec![];
    }
    if size == 1 {
        return vec![1.0];
    }
    
    (0..size)
        .map(|i| {
            let phase = 2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32;
            0.54 - 0.46 * phase.cos()
        })
        .collect()
}

/// Generate a rectangular (boxcar) window of the specified size
pub fn rectangular_window(size: usize) -> Vec<f32> {
    vec![1.0; size]
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hann_window() {
        let window = hann_window(4);
        assert_eq!(window.len(), 4);
        assert!((window[0] - 0.0).abs() < 1e-6); // First value should be ~0
        assert!((window[3] - 0.0).abs() < 1e-6); // Last value should be ~0
        assert!(window[1] > 0.0 && window[2] > 0.0); // Middle values should be positive
    }
}