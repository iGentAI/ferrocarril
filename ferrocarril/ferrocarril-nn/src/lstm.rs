//! Pure-Rust, dependency–free single–layer LSTM
//!  – handles batch_first / time-major layouts
//!  – supports optional (h0, c0) initial state
//!  – returns (Y, (hN, cN))
//!
//!  Shapes
//!  • batch_first = true
//!      input : [B, T, input_size]
//!      output: [B, T, hidden_size * (2 if bidirectional else 1)]
//!      hN, cN: [B, hidden_size * (2 if bidirectional else 1)]
//!
//!  • batch_first = false
//!      input : [T, B, input_size]
//!      output: [T, B, hidden_size * (2 if bidirectional else 1)]
//!      hN, cN: [B, hidden_size * (2 if bidirectional else 1)]

#![allow(dead_code)]

use crate::{Forward, Parameter};
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::FerroError;
use ferrocarril_core::weights::{PyTorchWeightLoader, LoadWeights};
#[cfg(feature = "weights")]
use ferrocarril_core::weights_binary::BinaryWeightLoader;
#[cfg(feature = "weights")]
use ferrocarril_core::LoadWeightsBinary;

/// Sigmoid activation
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
/// Tanh activation
#[inline(always)]
fn tanh(x: f32) -> f32 {
    x.tanh()
}

#[derive(Debug)]
pub struct LSTM {
    pub(crate) weight_ih_l0: Parameter,        // [4*H, input_size]
    pub(crate) weight_hh_l0: Parameter,        // [4*H, hidden_size]
    pub(crate) bias_ih_l0: Parameter,          // [4*H]
    pub(crate) bias_hh_l0: Parameter,          // [4*H]
    
    // New parameters for bidirectional support (reverse direction)
    pub(crate) weight_ih_l0_reverse: Parameter, // [4*H, input_size]
    pub(crate) weight_hh_l0_reverse: Parameter, // [4*H, hidden_size]
    pub(crate) bias_ih_l0_reverse: Parameter,   // [4*H]
    pub(crate) bias_hh_l0_reverse: Parameter,   // [4*H]

    // configuration
    input_size: usize,
    hidden_size: usize,  // hidden size per direction
    batch_first: bool,
    bidirectional: bool,
    num_layers:  usize,  // kept for future expansion
}

impl LSTM {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        batch_first: bool,
        bidirectional: bool,
    ) -> Self {
        assert_eq!(num_layers, 1, "only a single layer is supported");

        let gate_size = 4 * hidden_size;

        let weight_ih_l0 = Parameter::new(Tensor::new(vec![gate_size, input_size]));
        let weight_hh_l0 = Parameter::new(Tensor::new(vec![gate_size, hidden_size]));
        let bias_ih_l0 = Parameter::new(Tensor::new(vec![gate_size]));
        let bias_hh_l0 = Parameter::new(Tensor::new(vec![gate_size]));
        
        // Initialize reverse direction parameters (even if not bidirectional, for consistency)
        let weight_ih_l0_reverse = Parameter::new(Tensor::new(vec![gate_size, input_size]));
        let weight_hh_l0_reverse = Parameter::new(Tensor::new(vec![gate_size, hidden_size]));
        let bias_ih_l0_reverse = Parameter::new(Tensor::new(vec![gate_size]));
        let bias_hh_l0_reverse = Parameter::new(Tensor::new(vec![gate_size]));

        Self {
            weight_ih_l0,
            weight_hh_l0,
            bias_ih_l0,
            bias_hh_l0,
            weight_ih_l0_reverse,
            weight_hh_l0_reverse,
            bias_ih_l0_reverse,
            bias_hh_l0_reverse,
            input_size,
            hidden_size,
            batch_first,
            bidirectional,
            num_layers,
        }
    }
    
    // Add a method to get the input size
    pub fn get_input_size(&self) -> usize {
        self.input_size
    }
    
    // Add a method to get the hidden size
    pub fn get_hidden_size(&self) -> usize {
        self.hidden_size
    }
    
    // Add a method to get the total output size accounting for bidirectionality
    pub fn get_output_size(&self) -> usize {
        self.hidden_size * if self.bidirectional { 2 } else { 1 }
    }

    // ---------------------------------------------------------------------
    // low-level helper working on a single sample of size (input_size)
    // ---------------------------------------------------------------------
    #[inline]
    fn step(
        &self,
        x_t: &[f32],          //  input  : &[input_size]
        h_t: &[f32],          //  prev h : &[hidden_size]
        c_t: &[f32],          //  prev c : &[hidden_size]
        gate_buf: &mut [f32], //  length = 4*hidden_size
        h_out: &mut [f32],    //  length = hidden_size
        c_out: &mut [f32],    //  length = hidden_size
        weights_ih: &Tensor<f32>, // [4*H, input_size]
        weights_hh: &Tensor<f32>, // [4*H, hidden_size]
        bias_ih: &Tensor<f32>,    // [4*H]
        bias_hh: &Tensor<f32>,    // [4*H]
    ) {
        debug_assert_eq!(x_t.len(), self.input_size);
        debug_assert_eq!(h_t.len(), self.hidden_size);
        debug_assert_eq!(c_t.len(), self.hidden_size);
        debug_assert_eq!(gate_buf.len(), 4 * self.hidden_size);

        let ih_data = weights_ih.data();
        let hh_data = weights_hh.data();
        let ih_bias = bias_ih.data();
        let hh_bias = bias_hh.data();

        let input_size = self.input_size;
        let hidden_size = self.hidden_size;
        let gate_total = 4 * hidden_size;

        // ---- gate = W_ih * x + W_hh * h + b_ih + b_hh --------------------
        for g in 0..gate_total {
            let w_ih_row = &ih_data[g * input_size..(g + 1) * input_size];
            let w_hh_row = &hh_data[g * hidden_size..(g + 1) * hidden_size];

            let mut sum = ih_bias[g] + hh_bias[g];

            for i in 0..input_size {
                sum += w_ih_row[i] * x_t[i];
            }
            for i in 0..hidden_size {
                sum += w_hh_row[i] * h_t[i];
            }

            gate_buf[g] = sum;
        }

        // ---- activations + state update ----------------------------------
        let h_size = self.hidden_size;
        for i in 0..h_size {
            let i_gate = sigmoid(gate_buf[i]);
            let f_gate = sigmoid(gate_buf[h_size + i]);
            let g_gate = tanh   (gate_buf[2 * h_size + i]);
            let o_gate = sigmoid(gate_buf[3 * h_size + i]);

            // new cell & hidden
            let c_new = f_gate * c_t[i] + i_gate * g_gate;
            let h_new = o_gate * tanh(c_new);

            c_out[i] = c_new;
            h_out[i] = h_new;
        }
    }

    // ---------------------------------------------------------------------
    // forward (batch_first = true)
    // ---------------------------------------------------------------------
    pub fn forward_batch_first(
        &self,
        input: &Tensor<f32>,
        h0: Option<&Tensor<f32>>,
        c0: Option<&Tensor<f32>>,
    ) -> (Tensor<f32>, (Tensor<f32>, Tensor<f32>)) {
        let shape = input.shape();
        
        // Validate input dimensions
        if shape.len() != 3 {
            panic!("Input tensor must be 3D [batch, seq_len, features], got shape: {:?}", shape);
        }
        
        let (_batch, _seq_len, feat) = (shape[0], shape[1], shape[2]);
        
        // Strict check for feature dimension match - this should panic if mismatched
        assert_eq!(feat, self.input_size, 
                  "Input feature dimension {} does not match LSTM input_size {}", 
                  feat, self.input_size);
        
        // Call the implementation method
        self.forward_batch_first_impl(input, h0, c0)
    }
    
    // Helper method to implement the actual forward pass
    fn forward_batch_first_impl(
        &self,
        input: &Tensor<f32>,
        h0: Option<&Tensor<f32>>,
        c0: Option<&Tensor<f32>>,
    ) -> (Tensor<f32>, (Tensor<f32>, Tensor<f32>)) {
        let shape = input.shape();
        let (batch, seq_len, feat) = (shape[0], shape[1], shape[2]);
        
        // Calculate output dimension based on bidirectionality
        let output_size = self.hidden_size * (if self.bidirectional { 2 } else { 1 });

        // Initial hidden and cell states
        let mut h_state_fwd = if let Some(h_tensor) = h0 {
            // If provided h0 for bidirectional mode, we need to split it
            if self.bidirectional && h_tensor.shape()[1] == output_size {
                let mut h_fwd = vec![0.0; batch * self.hidden_size];
                for b in 0..batch {
                    for idx in 0..self.hidden_size {
                        h_fwd[b * self.hidden_size + idx] = h_tensor[&[b, idx]];
                    }
                }
                h_fwd
            } else {
                // For unidirectional or already split h0
                h_tensor.data().to_vec()
            }
        } else {
            vec![0.0; batch * self.hidden_size]
        };
        
        let mut c_state_fwd = if let Some(c_tensor) = c0 {
            // If provided c0 for bidirectional mode, we need to split it
            if self.bidirectional && c_tensor.shape()[1] == output_size {
                let mut c_fwd = vec![0.0; batch * self.hidden_size];
                for b in 0..batch {
                    for idx in 0..self.hidden_size {
                        c_fwd[b * self.hidden_size + idx] = c_tensor[&[b, idx]];
                    }
                }
                c_fwd
            } else {
                // For unidirectional or already split c0
                c_tensor.data().to_vec()
            }
        } else {
            vec![0.0; batch * self.hidden_size]
        };
        
        // For bidirectional LSTM, we also need backward states
        let mut h_state_bwd = if self.bidirectional {
            if let Some(h_tensor) = h0 {
                if h_tensor.shape()[1] == output_size {
                    let mut h_bwd = vec![0.0; batch * self.hidden_size];
                    for b in 0..batch {
                        for idx in 0..self.hidden_size {
                            h_bwd[b * self.hidden_size + idx] = h_tensor[&[b, idx + self.hidden_size]];
                        }
                    }
                    h_bwd
                } else {
                    vec![0.0; batch * self.hidden_size]
                }
            } else {
                vec![0.0; batch * self.hidden_size]
            }
        } else {
            vec![0.0; batch * self.hidden_size]
        };
        
        let mut c_state_bwd = if self.bidirectional {
            if let Some(c_tensor) = c0 {
                if c_tensor.shape()[1] == output_size {
                    let mut c_bwd = vec![0.0; batch * self.hidden_size];
                    for b in 0..batch {
                        for idx in 0..self.hidden_size {
                            c_bwd[b * self.hidden_size + idx] = c_tensor[&[b, idx + self.hidden_size]];
                        }
                    }
                    c_bwd
                } else {
                    vec![0.0; batch * self.hidden_size]
                }
            } else {
                vec![0.0; batch * self.hidden_size]
            }
        } else {
            vec![0.0; batch * self.hidden_size]
        };

        // Buffers reused per sample
        let mut gate_buf = vec![0.0f32; 4 * self.hidden_size];
        let mut h_tmp    = vec![0.0f32; self.hidden_size];
        let mut c_tmp    = vec![0.0f32; self.hidden_size];
        
        // Output tensors
        let mut y_fwd = vec![0.0f32; batch * seq_len * self.hidden_size];
        let mut y_bwd = if self.bidirectional {
            vec![0.0f32; batch * seq_len * self.hidden_size]
        } else {
            vec![]
        };

        // Forward pass
        for t in 0..seq_len {
            for b in 0..batch {
                // Slice helpers
                let x_offset = (b * seq_len + t) * feat;
                let h_offset = b * self.hidden_size;
                
                let x_t = &input.data()[x_offset..x_offset + feat];
                let h_prev = &h_state_fwd[h_offset..h_offset + self.hidden_size];
                let c_prev = &c_state_fwd[h_offset..h_offset + self.hidden_size];

                // Forward direction
                self.step(
                    x_t, h_prev, c_prev, &mut gate_buf, &mut h_tmp, &mut c_tmp,
                    self.weight_ih_l0.data(), self.weight_hh_l0.data(),
                    self.bias_ih_l0.data(), self.bias_hh_l0.data(),
                );

                // Update hidden and cell states
                h_state_fwd[h_offset..h_offset + self.hidden_size]
                    .copy_from_slice(&h_tmp);
                c_state_fwd[h_offset..h_offset + self.hidden_size]
                    .copy_from_slice(&c_tmp);

                // Write output Y for forward direction
                let y_offset = (b * seq_len + t) * self.hidden_size;
                y_fwd[y_offset..y_offset + self.hidden_size]
                    .copy_from_slice(&h_tmp);
            }
        }
        
        // Backward pass for bidirectional LSTM
        if self.bidirectional {
            for t in (0..seq_len).rev() {  // Process in reverse order
                for b in 0..batch {
                    // Slice helpers
                    let x_offset = (b * seq_len + t) * feat;
                    let h_offset = b * self.hidden_size;
                    
                    let x_t = &input.data()[x_offset..x_offset + feat];
                    let h_prev = &h_state_bwd[h_offset..h_offset + self.hidden_size];
                    let c_prev = &c_state_bwd[h_offset..h_offset + self.hidden_size];
                    
                    // Backward direction using reverse weights
                    self.step(
                        x_t, h_prev, c_prev, &mut gate_buf, &mut h_tmp, &mut c_tmp,
                        self.weight_ih_l0_reverse.data(), self.weight_hh_l0_reverse.data(),
                        self.bias_ih_l0_reverse.data(), self.bias_hh_l0_reverse.data(),
                    );
                    
                    // Update hidden and cell states for backward direction
                    h_state_bwd[h_offset..h_offset + self.hidden_size]
                        .copy_from_slice(&h_tmp);
                    c_state_bwd[h_offset..h_offset + self.hidden_size]
                        .copy_from_slice(&c_tmp);
                    
                    // Write output Y for backward direction
                    let y_offset = (b * seq_len + t) * self.hidden_size;
                    y_bwd[y_offset..y_offset + self.hidden_size]
                        .copy_from_slice(&h_tmp);
                }
            }
        }
        
        // Combine outputs for forward and backward directions
        let mut y_combined;
        let mut h_combined;
        let mut c_combined;
        
        if self.bidirectional {
            // Create concatenated outputs [B, T, 2*hidden_size]
            y_combined = vec![0.0; batch * seq_len * output_size];
            
            for b in 0..batch {
                for t in 0..seq_len {
                    for idx in 0..self.hidden_size {
                        // Forward part
                        let fwd_idx = (b * seq_len + t) * self.hidden_size + idx;
                        let combined_idx = (b * seq_len + t) * output_size + idx;
                        y_combined[combined_idx] = y_fwd[fwd_idx];
                        
                        // Backward part
                        let bwd_idx = (b * seq_len + t) * self.hidden_size + idx;
                        let combined_idx = (b * seq_len + t) * output_size + self.hidden_size + idx;
                        y_combined[combined_idx] = y_bwd[bwd_idx];
                    }
                }
            }
            
            // Add validation assertion after concatenation
            assert_eq!(output_size, self.hidden_size * 2,
                "For bidirectional LSTM, output_size must be 2x hidden_size. Got {} output size but expected {}",
                output_size, self.hidden_size * 2);
            
            // Combine final hidden and cell states
            h_combined = vec![0.0; batch * output_size];
            c_combined = vec![0.0; batch * output_size];
            
            for b in 0..batch {
                for idx in 0..self.hidden_size {
                    // Forward part
                    h_combined[b * output_size + idx] = h_state_fwd[b * self.hidden_size + idx];
                    c_combined[b * output_size + idx] = c_state_fwd[b * self.hidden_size + idx];
                    
                    // Backward part
                    h_combined[b * output_size + self.hidden_size + idx] = h_state_bwd[b * self.hidden_size + idx];
                    c_combined[b * output_size + self.hidden_size + idx] = c_state_bwd[b * self.hidden_size + idx];
                }
            }
        } else {
            // For unidirectional, just use forward outputs
            y_combined = y_fwd;
            h_combined = h_state_fwd;
            c_combined = c_state_fwd;
        }

        // Return final tensors
        (
            Tensor::from_data(y_combined, vec![batch, seq_len, output_size]),
            (
                Tensor::from_data(h_combined, vec![batch, output_size]),
                Tensor::from_data(c_combined, vec![batch, output_size]),
            ),
        )
    }

    // ---------------------------------------------------------------------
    // time-major helper (batch_first = false)
    // ---------------------------------------------------------------------
    pub fn forward_time_major(
        &self,
        input: &Tensor<f32>,
        h0: Option<&Tensor<f32>>,
        c0: Option<&Tensor<f32>>,
    ) -> (Tensor<f32>, (Tensor<f32>, Tensor<f32>)) {
        let shape = input.shape();
        if shape.len() != 3 {
            panic!("Input tensor must be 3D [time, batch, features], got shape: {:?}", shape);
        }
        
        let (seq_len, batch, feat) = (shape[0], shape[1], shape[2]);
        
        // Convert time-major to batch-first for unified processing.
        // (Dimension mismatches are asserted inside forward_batch_first.)
        let mut input_batch_first_data = vec![0.0; batch * seq_len * feat];
        for t in 0..seq_len {
            for b in 0..batch {
                for f in 0..feat {
                    input_batch_first_data[(b * seq_len + t) * feat + f] = 
                        input.data()[(t * batch + b) * feat + f];
                }
            }
        }
        let input_batch_first = Tensor::from_data(input_batch_first_data, vec![batch, seq_len, feat]);
        
        // Process using the batch-first implementation
        // We'll use forward_batch_first instead of _impl directly so it can adapt dimensions if needed
        let (output_batch_first, (h_final, c_final)) = 
            self.forward_batch_first(&input_batch_first, h0, c0);
        
        // Convert output back to time-major format
        let output_size = output_batch_first.shape()[2];
        let mut output_time_major_data = vec![0.0; seq_len * batch * output_size];
        for b in 0..batch {
            for t in 0..seq_len {
                for o in 0..output_size {
                    output_time_major_data[(t * batch + b) * output_size + o] = 
                        output_batch_first.data()[(b * seq_len + t) * output_size + o];
                }
            }
        }
        
        (
            Tensor::from_data(output_time_major_data, vec![seq_len, batch, output_size]),
            (h_final, c_final),
        )
    }

    // ---------------------------------------------------------------------
    // public convenience forward for *unbatched* input
    // ---------------------------------------------------------------------
    pub fn forward_unbatched(
        &self,
        input: &Tensor<f32>,
        hidden: Option<&Tensor<f32>>,
        cell:   Option<&Tensor<f32>>,
    ) -> (Tensor<f32>, (Tensor<f32>, Tensor<f32>)) {
        // treat a [T, F] tensor as [1, T, F]
        let seq_len = input.shape()[0];
        let reshaped = input.reshape(&[1, seq_len, self.input_size]);
        let (y, (h, c)) = self.forward_batch_first(&reshaped, hidden, cell);
        
        // Determine the output hidden size
        let output_hidden_size = y.shape()[2];
        
        // strip the batch dim again for the return value
        (
            y.reshape(&[seq_len, output_hidden_size]),
            (
                h.reshape(&[output_hidden_size]),
                c.reshape(&[output_hidden_size]),
            ),
        )
    }
    
    /// Helper method for loading weights with reverse flag
    #[cfg(feature = "weights")]
    pub fn load_weights_binary_with_reverse(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
        is_reverse: bool
    ) -> Result<(), FerroError> {
        // Determine suffix based on whether this is a reverse LSTM
        let suffix = if is_reverse { "_reverse" } else { "" };
        
        // Choose the right parameters to load into based on direction
        let (weight_ih, weight_hh, bias_ih, bias_hh) = if is_reverse {
            (
                &mut self.weight_ih_l0_reverse,
                &mut self.weight_hh_l0_reverse,
                &mut self.bias_ih_l0_reverse,
                &mut self.bias_hh_l0_reverse
            )
        } else {
            (
                &mut self.weight_ih_l0,
                &mut self.weight_hh_l0,
                &mut self.bias_ih_l0,
                &mut self.bias_hh_l0
            )
        };
        
        // Load weight_ih_l0
        let weight_ih_path = format!("{}.weight_ih_l0{}", prefix, suffix);
        match loader.load_component_parameter(component, &weight_ih_path) {
            Ok(tensor) => {
                // Validate shape
                if tensor.shape()[0] != 4 * self.hidden_size || tensor.shape()[1] != self.input_size {
                    return Err(FerroError::new(format!(
                        "Invalid weight_ih shape: {:?}, expected [{}, {}]", 
                        tensor.shape(), 4 * self.hidden_size, self.input_size
                    )));
                }
                *weight_ih = Parameter::new(tensor);
            },
            Err(err) => {
                return Err(FerroError::new(format!("Failed to load {}: {}", weight_ih_path, err)));
            }
        }
        
        // Load weight_hh_l0
        let weight_hh_path = format!("{}.weight_hh_l0{}", prefix, suffix);
        match loader.load_component_parameter(component, &weight_hh_path) {
            Ok(tensor) => {
                // Validate shape
                if tensor.shape()[0] != 4 * self.hidden_size || tensor.shape()[1] != self.hidden_size {
                    return Err(FerroError::new(format!(
                        "Invalid weight_hh shape: {:?}, expected [{}, {}]", 
                        tensor.shape(), 4 * self.hidden_size, self.hidden_size
                    )));
                }
                *weight_hh = Parameter::new(tensor);
            },
            Err(err) => {
                return Err(FerroError::new(format!("Failed to load {}: {}", weight_hh_path, err)));
            }
        }
        
        // Load bias_ih_l0
        let bias_ih_path = format!("{}.bias_ih_l0{}", prefix, suffix);
        match loader.load_component_parameter(component, &bias_ih_path) {
            Ok(tensor) => {
                // Validate shape
                if tensor.shape()[0] != 4 * self.hidden_size {
                    return Err(FerroError::new(format!(
                        "Invalid bias_ih shape: {:?}, expected [{}]", 
                        tensor.shape(), 4 * self.hidden_size
                    )));
                }
                *bias_ih = Parameter::new(tensor);
            },
            Err(err) => {
                return Err(FerroError::new(format!("Failed to load {}: {}", bias_ih_path, err)));
            }
        }
        
        // Load bias_hh_l0
        let bias_hh_path = format!("{}.bias_hh_l0{}", prefix, suffix);
        match loader.load_component_parameter(component, &bias_hh_path) {
            Ok(tensor) => {
                // Validate shape
                if tensor.shape()[0] != 4 * self.hidden_size {
                    return Err(FerroError::new(format!(
                        "Invalid bias_hh shape: {:?}, expected [{}]", 
                        tensor.shape(), 4 * self.hidden_size
                    )));
                }
                *bias_hh = Parameter::new(tensor);
            },
            Err(err) => {
                return Err(FerroError::new(format!("Failed to load {}: {}", bias_hh_path, err)));
            }
        }
        
        Ok(())
    }
}

#[cfg(feature = "weights")]
impl LoadWeightsBinary for LSTM {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str
    ) -> Result<(), FerroError> {
        // Load forward direction weights
        let mut errors = Vec::new();
        
        match self.load_weights_binary_with_reverse(loader, component, prefix, false) {
            Ok(_) => {},
            Err(e) => errors.push(format!("Failed to load forward weights: {}", e)),
        }
        
        // Load reverse direction weights if bidirectional
        if self.bidirectional {
            match self.load_weights_binary_with_reverse(loader, component, prefix, true) {
                Ok(_) => {},
                Err(e) => errors.push(format!("Failed to load reverse weights: {}", e)),
            }
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(FerroError::new(errors.join("; ")))
        }
    }
}

impl LoadWeights for LSTM {
    fn load_weights(
        &mut self, 
        loader: &PyTorchWeightLoader, 
        prefix: Option<&str>
    ) -> Result<(), FerroError> {
        // Load forward direction weights
        loader.load_weight_into_parameter(&mut self.weight_ih_l0, "weight_ih_l0", prefix, None)?;
        loader.load_weight_into_parameter(&mut self.weight_hh_l0, "weight_hh_l0", prefix, None)?;
        loader.load_weight_into_parameter(&mut self.bias_ih_l0, "bias_ih_l0", prefix, None)?;
        loader.load_weight_into_parameter(&mut self.bias_hh_l0, "bias_hh_l0", prefix, None)?;
        
        // Load reverse direction weights if bidirectional
        if self.bidirectional {
            loader.load_weight_into_parameter(&mut self.weight_ih_l0_reverse, "weight_ih_l0_reverse", prefix, None)?;
            loader.load_weight_into_parameter(&mut self.weight_hh_l0_reverse, "weight_hh_l0_reverse", prefix, None)?;
            loader.load_weight_into_parameter(&mut self.bias_ih_l0_reverse, "bias_ih_l0_reverse", prefix, None)?;
            loader.load_weight_into_parameter(&mut self.bias_hh_l0_reverse, "bias_hh_l0_reverse", prefix, None)?;
        }
        
        Ok(())
    }
}

// -------------------------------------------------------------------------
// Forward trait implementation
// -------------------------------------------------------------------------
impl Forward for LSTM {
    type Output = (Tensor<f32>, (Tensor<f32>, Tensor<f32>));

    fn forward(&self, input: &Tensor<f32>) -> Self::Output {
        if self.batch_first {
            self.forward_batch_first(input, None, None)
        } else {
            self.forward_time_major(input, None, None)
        }
    }
}

// -------------------------------------------------------------------------
// tiny verification
// -------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_unbatched() {
        let lstm = LSTM::new(8, 4, 1, false, false);
        // input [T, F] with T = 3
        let inp = Tensor::new(vec![3, 8]);
        let (y, (h, c)) = lstm.forward(&inp);
        assert_eq!(y.shape(), &[3, 4]);
        assert_eq!(h.shape(), &[4]);
        assert_eq!(c.shape(), &[4]);
    }

    #[test]
    fn smoke_batch() {
        let lstm = LSTM::new(10, 6, 1, true, false);
        // input [B, T, F]  (B = 2, T = 5)
        let inp = Tensor::new(vec![2, 5, 10]);
        let (y, (h, c)) = lstm.forward(&inp);
        assert_eq!(y.shape(), &[2, 5, 6]);
        assert_eq!(h.shape(), &[2, 6]);
        assert_eq!(c.shape(), &[2, 6]);
    }
    
    #[test]
    fn smoke_bidirectional() {
        // Test that bidirectional mode works correctly
        let lstm = LSTM::new(10, 6, 1, true, true);
        // input [B, T, F]  (B = 2, T = 5)
        let inp = Tensor::new(vec![2, 5, 10]);
        let (y, (h, c)) = lstm.forward(&inp);
        // For bidirectional, output hidden size is doubled
        assert_eq!(y.shape(), &[2, 5, 12]);
        assert_eq!(h.shape(), &[2, 12]);
        assert_eq!(c.shape(), &[2, 12]);
    }
}