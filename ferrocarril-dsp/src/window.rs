//! Window functions for DSP operations

use std::f32::consts::PI;

/// Create a Hann window
pub fn hann_window(size: usize, periodic: bool) -> Vec<f32> {
    let mut window = vec![0.0; size];
    // For periodic: n = size
    // For non-periodic: n = size - 1
    let n = if periodic { size as f32 } else { (size - 1) as f32 };
    
    for i in 0..size {
        window[i] = 0.5 * (1.0 - (2.0 * PI * i as f32 / n).cos());
    }
    
    window
}

/// Create a Hamming window
pub fn hamming_window(size: usize, periodic: bool) -> Vec<f32> {
    let mut window = vec![0.0; size];
    // For periodic: n = size
    // For non-periodic: n = size - 1
    let n = if periodic { size as f32 } else { (size - 1) as f32 };
    
    for i in 0..size {
        window[i] = 0.54 - 0.46 * (2.0 * PI * i as f32 / n).cos();
    }
    
    window
}

/// Create a rectangle window
pub fn rect_window(size: usize) -> Vec<f32> {
    vec![1.0; size]
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hann_window() {
        let window = hann_window(4, false);
        assert_eq!(window.len(), 4);
        // For non-periodic hann window with size 4 and n=3:
        // i=0: 0.5 * (1 - cos(0)) = 0.5 * 0 = 0
        // i=2: 0.5 * (1 - cos(2*PI*2/3)) = 0.5 * (1 - (-0.5)) = 0.75
        assert!((window[0] - 0.0).abs() < 1e-6);
        assert!((window[2] - 0.75).abs() < 1e-6);
    }
    
    #[test]
    fn test_periodic_hann_window() {
        let window = hann_window(4, true);
        assert_eq!(window.len(), 4);
        // For periodic hann window with size 4 and n=4:
        // i=0: 0.5 * (1 - cos(0)) = 0.5 * 0 = 0
        // i=1: 0.5 * (1 - cos(2*PI*1/4)) = 0.5 * (1 - 0) = 0.5
        // i=2: 0.5 * (1 - cos(2*PI*2/4)) = 0.5 * (1 - (-1)) = 1.0
        // i=3: 0.5 * (1 - cos(2*PI*3/4)) = 0.5 * (1 - 0) = 0.5
        assert!((window[0] - 0.0).abs() < 1e-6);
        assert!((window[1] - 0.5).abs() < 1e-6);
        assert!((window[2] - 1.0).abs() < 1e-6);
        assert!((window[3] - 0.5).abs() < 1e-6);
    }
}