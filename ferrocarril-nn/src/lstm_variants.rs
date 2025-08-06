//! Specialized LSTM implementations for different Kokoro TTS components
//! 
//! Each LSTM variant is designed for a specific use case in the Kokoro pipeline:
//! - TextEncoderLSTM: 512→512, final phoneme encoding
//! - ProsodyLSTM: 640→512, style-conditioned duration/F0 prediction  
//! - DurationEncoderLSTM: 640→512, multi-layer duration encoding
//!
//! All variants handle PyTorch's stacked IFGO gate weights and bidirectional processing.

use crate::Parameter;
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::{FerroError, LoadWeightsBinary};
use ferrocarril_core::weights_binary::BinaryWeightLoader;

/// LSTM gate enumeration matching PyTorch's storage order
#[derive(Debug, Clone, Copy)]
enum LSTMGate {
    Input = 0,   // Input gate
    Forget = 1,  // Forget gate  
    Cell = 2,    // Cell gate (g)
    Output = 3,  // Output gate
}

/// Helper function to extract gate weights from PyTorch's stacked format
/// 
/// PyTorch stores LSTM weights as [4*hidden_size, input_size] where the 4x factor
/// represents IFGO gates concatenated. This function splits them.
fn extract_gate_weights(
    stacked_weight: &Tensor<f32>, 
    gate: LSTMGate, 
    hidden_size: usize
) -> Tensor<f32> {
    let input_size = stacked_weight.shape()[1];
    let gate_start = gate as usize * hidden_size;
    let gate_end = gate_start + hidden_size;
    
    // Extract the gate's weight slice
    let mut gate_data = vec![0.0; hidden_size * input_size];
    
    for h in 0..hidden_size {
        for i in 0..input_size {
            gate_data[h * input_size + i] = stacked_weight[&[gate_start + h, i]];
        }
    }
    
    Tensor::from_data(gate_data, vec![hidden_size, input_size])
}

/// Extract gate biases from PyTorch's stacked format
fn extract_gate_biases(stacked_bias: &Tensor<f32>, gate: LSTMGate, hidden_size: usize) -> Tensor<f32> {
    let gate_start = gate as usize * hidden_size;
    let gate_end = gate_start + hidden_size;
    
    let gate_data = stacked_bias.data()[gate_start..gate_end].to_vec();
    Tensor::from_data(gate_data, vec![hidden_size])
}

/// LSTM gates structure for one direction
#[derive(Debug)]
struct LSTMGates {
    input_weight: Parameter,   // [hidden_size, input_size]
    forget_weight: Parameter,  // [hidden_size, input_size]
    cell_weight: Parameter,    // [hidden_size, input_size]
    output_weight: Parameter,  // [hidden_size, input_size]
    
    input_recurrent: Parameter,   // [hidden_size, hidden_size]
    forget_recurrent: Parameter,  // [hidden_size, hidden_size]
    cell_recurrent: Parameter,    // [hidden_size, hidden_size]
    output_recurrent: Parameter,  // [hidden_size, hidden_size]
    
    input_bias: Parameter,     // [hidden_size]
    forget_bias: Parameter,    // [hidden_size]
    cell_bias: Parameter,      // [hidden_size]
    output_bias: Parameter,    // [hidden_size]
}

impl LSTMGates {
    fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            input_weight: Parameter::new(Tensor::new(vec![hidden_size, input_size])),
            forget_weight: Parameter::new(Tensor::new(vec![hidden_size, input_size])),
            cell_weight: Parameter::new(Tensor::new(vec![hidden_size, input_size])),
            output_weight: Parameter::new(Tensor::new(vec![hidden_size, input_size])),
            
            input_recurrent: Parameter::new(Tensor::new(vec![hidden_size, hidden_size])),
            forget_recurrent: Parameter::new(Tensor::new(vec![hidden_size, hidden_size])),
            cell_recurrent: Parameter::new(Tensor::new(vec![hidden_size, hidden_size])),
            output_recurrent: Parameter::new(Tensor::new(vec![hidden_size, hidden_size])),
            
            input_bias: Parameter::new(Tensor::new(vec![hidden_size])),
            forget_bias: Parameter::new(Tensor::new(vec![hidden_size])),
            cell_bias: Parameter::new(Tensor::new(vec![hidden_size])),
            output_bias: Parameter::new(Tensor::new(vec![hidden_size])),
        }
    }
    
    /// Load gate weights from PyTorch's stacked format
    fn load_from_stacked_weights(
        &mut self,
        weight_ih: &Tensor<f32>,  // [4*hidden_size, input_size]
        weight_hh: &Tensor<f32>,  // [4*hidden_size, hidden_size]
        bias_ih: &Tensor<f32>,    // [4*hidden_size]
        bias_hh: &Tensor<f32>,    // [4*hidden_size]
        hidden_size: usize,
    ) -> Result<(), FerroError> {
        // Extract input-hidden weights (IFGO order)
        self.input_weight = Parameter::new(extract_gate_weights(weight_ih, LSTMGate::Input, hidden_size));
        self.forget_weight = Parameter::new(extract_gate_weights(weight_ih, LSTMGate::Forget, hidden_size));
        self.cell_weight = Parameter::new(extract_gate_weights(weight_ih, LSTMGate::Cell, hidden_size));
        self.output_weight = Parameter::new(extract_gate_weights(weight_ih, LSTMGate::Output, hidden_size));
        
        // Extract hidden-hidden weights (IFGO order)
        self.input_recurrent = Parameter::new(extract_gate_weights(weight_hh, LSTMGate::Input, hidden_size));
        self.forget_recurrent = Parameter::new(extract_gate_weights(weight_hh, LSTMGate::Forget, hidden_size));
        self.cell_recurrent = Parameter::new(extract_gate_weights(weight_hh, LSTMGate::Cell, hidden_size));
        self.output_recurrent = Parameter::new(extract_gate_weights(weight_hh, LSTMGate::Output, hidden_size));
        
        // Extract biases
        let ih_input_bias = extract_gate_biases(bias_ih, LSTMGate::Input, hidden_size);
        let ih_forget_bias = extract_gate_biases(bias_ih, LSTMGate::Forget, hidden_size);
        let ih_cell_bias = extract_gate_biases(bias_ih, LSTMGate::Cell, hidden_size);
        let ih_output_bias = extract_gate_biases(bias_ih, LSTMGate::Output, hidden_size);
        
        // Extract hidden-hidden biases (PyTorch convention)
        let hh_input_bias = extract_gate_biases(bias_hh, LSTMGate::Input, hidden_size);
        let hh_forget_bias = extract_gate_biases(bias_hh, LSTMGate::Forget, hidden_size);
        let hh_cell_bias = extract_gate_biases(bias_hh, LSTMGate::Cell, hidden_size);
        let hh_output_bias = extract_gate_biases(bias_hh, LSTMGate::Output, hidden_size);
        
        // Combine biases safely without unsafe mutations
        let mut input_bias_data = ih_input_bias.data().to_vec();
        let mut forget_bias_data = ih_forget_bias.data().to_vec();
        let mut cell_bias_data = ih_cell_bias.data().to_vec();
        let mut output_bias_data = ih_output_bias.data().to_vec();
        
        for h in 0..hidden_size {
            input_bias_data[h] += hh_input_bias[&[h]];
            forget_bias_data[h] += hh_forget_bias[&[h]];
            cell_bias_data[h] += hh_cell_bias[&[h]];
            output_bias_data[h] += hh_output_bias[&[h]];
        }
        
        // Create new parameter tensors with combined biases
        self.input_bias = Parameter::new(Tensor::from_data(input_bias_data, vec![hidden_size]));
        self.forget_bias = Parameter::new(Tensor::from_data(forget_bias_data, vec![hidden_size]));
        self.cell_bias = Parameter::new(Tensor::from_data(cell_bias_data, vec![hidden_size]));
        self.output_bias = Parameter::new(Tensor::from_data(output_bias_data, vec![hidden_size]));
        
        Ok(())
    }
    
    /// Compute LSTM gates for a single timestep
    fn compute_timestep(
        &self,
        input_t: &[f32],      // [input_size]
        hidden_t: &[f32],     // [hidden_size]
        cell_t: &[f32],       // [hidden_size]
    ) -> (Vec<f32>, Vec<f32>) {  // (new_hidden, new_cell)
        let hidden_size = cell_t.len();
        let input_size = input_t.len();
        
        // Compute gate activations
        let mut input_gate = vec![0.0; hidden_size];
        let mut forget_gate = vec![0.0; hidden_size];
        let mut cell_gate = vec![0.0; hidden_size];
        let mut output_gate = vec![0.0; hidden_size];
        
        for h in 0..hidden_size {
            // Input gate: σ(W_ii @ x + W_hi @ h + b_i)
            input_gate[h] = self.input_bias.data()[&[h]];
            for i in 0..input_size {
                input_gate[h] += self.input_weight.data()[&[h, i]] * input_t[i];
            }
            for prev_h in 0..hidden_size {
                input_gate[h] += self.input_recurrent.data()[&[h, prev_h]] * hidden_t[prev_h];
            }
            input_gate[h] = sigmoid(input_gate[h]);
            
            // Forget gate: σ(W_if @ x + W_hf @ h + b_f)
            forget_gate[h] = self.forget_bias.data()[&[h]];
            for i in 0..input_size {
                forget_gate[h] += self.forget_weight.data()[&[h, i]] * input_t[i];
            }
            for prev_h in 0..hidden_size {
                forget_gate[h] += self.forget_recurrent.data()[&[h, prev_h]] * hidden_t[prev_h];
            }
            forget_gate[h] = sigmoid(forget_gate[h]);
            
            // Cell gate: tanh(W_ic @ x + W_hc @ h + b_c)
            cell_gate[h] = self.cell_bias.data()[&[h]];
            for i in 0..input_size {
                cell_gate[h] += self.cell_weight.data()[&[h, i]] * input_t[i];
            }
            for prev_h in 0..hidden_size {
                cell_gate[h] += self.cell_recurrent.data()[&[h, prev_h]] * hidden_t[prev_h];
            }
            cell_gate[h] = cell_gate[h].tanh();
            
            // Output gate: σ(W_io @ x + W_ho @ h + b_o)
            output_gate[h] = self.output_bias.data()[&[h]];
            for i in 0..input_size {
                output_gate[h] += self.output_weight.data()[&[h, i]] * input_t[i];
            }
            for prev_h in 0..hidden_size {
                output_gate[h] += self.output_recurrent.data()[&[h, prev_h]] * hidden_t[prev_h];
            }
            output_gate[h] = sigmoid(output_gate[h]);
        }
        
        // Update cell and hidden states
        let mut new_cell = vec![0.0; hidden_size];
        let mut new_hidden = vec![0.0; hidden_size];
        
        for h in 0..hidden_size {
            new_cell[h] = forget_gate[h] * cell_t[h] + input_gate[h] * cell_gate[h];
            new_hidden[h] = output_gate[h] * new_cell[h].tanh();
        }
        
        (new_hidden, new_cell)
    }
}

/// TextEncoder LSTM: 512→512 bidirectional encoding for phoneme sequences
#[derive(Debug)]
pub struct TextEncoderLSTM {
    forward_gates: LSTMGates,
    reverse_gates: LSTMGates,
    input_size: usize,    // 512 - TextEncoder channels
    hidden_size: usize,   // 256 - half of output (bidirectional doubles it)
}

impl TextEncoderLSTM {
    /// Create new TextEncoder LSTM
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        assert_eq!(input_size, 512, "TextEncoder LSTM must have input_size=512");
        assert_eq!(hidden_size, 256, "TextEncoder LSTM must have hidden_size=256");
        
        Self {
            forward_gates: LSTMGates::new(input_size, hidden_size),
            reverse_gates: LSTMGates::new(input_size, hidden_size),
            input_size,
            hidden_size,
        }
    }
    
    /// Forward pass with pack_padded_sequence behavior
    /// 
    /// Input: [batch_size, seq_length, input_size]
    /// Output: [batch_size, seq_length, 2*hidden_size] (bidirectional)
    pub fn forward_batch_first_with_lengths(
        &self,
        input: &Tensor<f32>,
        input_lengths: &[usize],
        _initial_hidden: Option<&Tensor<f32>>,
        _initial_cell: Option<&Tensor<f32>>,
    ) -> (Tensor<f32>, (Tensor<f32>, Tensor<f32>)) {
        // STRICT: Validate input shape
        assert_eq!(input.shape().len(), 3,
            "STRICT: TextEncoderLSTM input must be 3D [batch, time, features], got: {:?}", input.shape());
        assert_eq!(input.shape()[2], self.input_size,
            "STRICT: TextEncoderLSTM input features {} must equal {}", 
            input.shape()[2], self.input_size);
            
        let (batch_size, seq_length, _) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        
        // STRICT: Validate input_lengths
        assert_eq!(input_lengths.len(), batch_size,
            "STRICT: input_lengths count {} must match batch_size {}", 
            input_lengths.len(), batch_size);
            
        let output_size = 2 * self.hidden_size; // Bidirectional
        let mut output_data = vec![0.0; batch_size * seq_length * output_size];
        
        // Variables to track final states for return
        let mut final_forward_hidden = vec![0.0; self.hidden_size];
        let mut final_forward_cell = vec![0.0; self.hidden_size];
        
        // Process each sequence in the batch
        for b in 0..batch_size {
            let actual_length = input_lengths[b].min(seq_length);
            
            // Forward direction
            let mut forward_hidden = vec![0.0; self.hidden_size];
            let mut forward_cell = vec![0.0; self.hidden_size];
            let mut forward_outputs = vec![vec![0.0; self.hidden_size]; actual_length];
            
            for t in 0..actual_length {
                // Extract input at timestep t
                let mut input_t = vec![0.0; self.input_size];
                for i in 0..self.input_size {
                    input_t[i] = input[&[b, t, i]];
                }
                
                let (new_hidden, new_cell) = self.forward_gates.compute_timestep(
                    &input_t, &forward_hidden, &forward_cell
                );
                
                forward_hidden = new_hidden.clone();
                forward_cell = new_cell;
                forward_outputs[t] = new_hidden.clone();
            }
            
            // Save final states from last batch item
            if b == batch_size - 1 {
                final_forward_hidden = forward_hidden;
                final_forward_cell = forward_cell;
            }
            
            // Reverse direction  
            let mut reverse_hidden = vec![0.0; self.hidden_size];
            let mut reverse_cell = vec![0.0; self.hidden_size];
            let mut reverse_outputs = vec![vec![0.0; self.hidden_size]; actual_length];
            
            for t in (0..actual_length).rev() {
                // Extract input at timestep t  
                let mut input_t = vec![0.0; self.input_size];
                for i in 0..self.input_size {
                    input_t[i] = input[&[b, t, i]];
                }
                
                let (new_hidden, new_cell) = self.reverse_gates.compute_timestep(
                    &input_t, &reverse_hidden, &reverse_cell
                );
                
                reverse_hidden = new_hidden.clone();
                reverse_cell = new_cell;
                reverse_outputs[t] = new_hidden;
            }
            
            // Concatenate forward and reverse outputs
            for t in 0..actual_length {
                let output_base = b * seq_length * output_size + t * output_size;
                
                // Forward part [0:hidden_size]
                for h in 0..self.hidden_size {
                    output_data[output_base + h] = forward_outputs[t][h];
                }
                
                // Reverse part [hidden_size:2*hidden_size]
                for h in 0..self.hidden_size {
                    output_data[output_base + self.hidden_size + h] = reverse_outputs[t][h];
                }
            }
            
            // Zero out padding positions (t >= actual_length)
            for t in actual_length..seq_length {
                let output_base = b * seq_length * output_size + t * output_size;
                for h in 0..output_size {
                    output_data[output_base + h] = 0.0;
                }
            }
        }
        
        // Create output tensors
        let output = Tensor::from_data(output_data, vec![batch_size, seq_length, output_size]);
        let final_hidden = Tensor::from_data(final_forward_hidden, vec![self.hidden_size]);
        let final_cell = Tensor::from_data(final_forward_cell, vec![self.hidden_size]);
        
        (output, (final_hidden, final_cell))
    }
}

/// ProsodyLSTM: 640→512 for duration prediction and F0/noise conditioning
#[derive(Debug)]
pub struct ProsodyLSTM {
    forward_gates: LSTMGates,
    reverse_gates: LSTMGates,
    input_size: usize,    // 640 - hidden_dim + style_dim (512 + 128)
    hidden_size: usize,   // 256 - half of output (bidirectional doubles to 512)
}

impl ProsodyLSTM {
    /// Create new Prosody LSTM
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        assert_eq!(input_size, 640, "ProsodyLSTM must have input_size=640 (512+128)");
        assert_eq!(hidden_size, 256, "ProsodyLSTM must have hidden_size=256");
        
        Self {
            forward_gates: LSTMGates::new(input_size, hidden_size),
            reverse_gates: LSTMGates::new(input_size, hidden_size),
            input_size,
            hidden_size,
        }
    }
    
    /// Forward pass with sequence length handling
    pub fn forward_batch_first_with_lengths(
        &self,
        input: &Tensor<f32>,
        input_lengths: &[usize],
        _initial_hidden: Option<&Tensor<f32>>,
        _initial_cell: Option<&Tensor<f32>>,
    ) -> (Tensor<f32>, (Tensor<f32>, Tensor<f32>)) {
        // STRICT: Validate input for Prosody LSTM
        assert_eq!(input.shape().len(), 3,
            "STRICT: ProsodyLSTM input must be 3D [batch, time, features], got: {:?}", input.shape());
        assert_eq!(input.shape()[2], self.input_size,
            "STRICT: ProsodyLSTM input features {} must equal {} (512+128)", 
            input.shape()[2], self.input_size);
            
        let (batch_size, seq_length, _) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let output_size = 2 * self.hidden_size; // 512 total (256×2)
        
        // Similar bidirectional processing as TextEncoderLSTM
        let mut output_data = vec![0.0; batch_size * seq_length * output_size];
        
        for b in 0..batch_size {
            let actual_length = input_lengths[b].min(seq_length);
            
            // Forward pass
            let mut forward_hidden = vec![0.0; self.hidden_size];
            let mut forward_cell = vec![0.0; self.hidden_size];
            
            for t in 0..actual_length {
                let mut input_t = vec![0.0; self.input_size];
                for i in 0..self.input_size {
                    input_t[i] = input[&[b, t, i]];
                }
                
                let (new_hidden, new_cell) = self.forward_gates.compute_timestep(
                    &input_t, &forward_hidden, &forward_cell
                );
                forward_hidden = new_hidden;
                forward_cell = new_cell;
                
                // Store forward output
                let output_base = b * seq_length * output_size + t * output_size;
                for h in 0..self.hidden_size {
                    output_data[output_base + h] = forward_hidden[h];
                }
            }
            
            // Reverse pass
            let mut reverse_hidden = vec![0.0; self.hidden_size];
            let mut reverse_cell = vec![0.0; self.hidden_size];
            
            for t in (0..actual_length).rev() {
                let mut input_t = vec![0.0; self.input_size];
                for i in 0..self.input_size {
                    input_t[i] = input[&[b, t, i]];
                }
                
                let (new_hidden, new_cell) = self.reverse_gates.compute_timestep(
                    &input_t, &reverse_hidden, &reverse_cell
                );
                reverse_hidden = new_hidden;
                reverse_cell = new_cell;
                
                // Store reverse output  
                let output_base = b * seq_length * output_size + t * output_size;
                for h in 0..self.hidden_size {
                    output_data[output_base + self.hidden_size + h] = reverse_hidden[h];
                }
            }
        }
        
        let output = Tensor::from_data(output_data, vec![batch_size, seq_length, output_size]);
        let final_hidden = Tensor::from_data(vec![0.0; self.hidden_size], vec![self.hidden_size]);
        let final_cell = Tensor::from_data(vec![0.0; self.hidden_size], vec![self.hidden_size]);
        
        (output, (final_hidden, final_cell))
    }
}

/// Load weights for TextEncoderLSTM from Kokoro's text_encoder component
impl LoadWeightsBinary for TextEncoderLSTM {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        println!("Loading TextEncoderLSTM weights for {}.{}", component, prefix);
        
        // STRICT: Load forward direction weights (weight_ih_l0, weight_hh_l0, etc.)
        let weight_ih_fwd = loader.load_component_parameter(component, &format!("{}.weight_ih_l0", prefix))?;
        let weight_hh_fwd = loader.load_component_parameter(component, &format!("{}.weight_hh_l0", prefix))?;
        let bias_ih_fwd = loader.load_component_parameter(component, &format!("{}.bias_ih_l0", prefix))?;
        let bias_hh_fwd = loader.load_component_parameter(component, &format!("{}.bias_hh_l0", prefix))?;
        
        // STRICT: Validate shapes for TextEncoder (512 input features)
        assert_eq!(weight_ih_fwd.shape(), &[1024, 512],
            "STRICT: TextEncoder weight_ih_l0 must have shape [1024, 512], got {:?}", weight_ih_fwd.shape());
        assert_eq!(weight_hh_fwd.shape(), &[1024, 256], 
            "STRICT: TextEncoder weight_hh_l0 must have shape [1024, 256], got {:?}", weight_hh_fwd.shape());
            
        // Load forward gates
        self.forward_gates.load_from_stacked_weights(
            &weight_ih_fwd, &weight_hh_fwd, &bias_ih_fwd, &bias_hh_fwd, self.hidden_size
        )?;
        
        // STRICT: Load reverse direction weights  
        let weight_ih_rev = loader.load_component_parameter(component, &format!("{}.weight_ih_l0_reverse", prefix))?;
        let weight_hh_rev = loader.load_component_parameter(component, &format!("{}.weight_hh_l0_reverse", prefix))?;
        let bias_ih_rev = loader.load_component_parameter(component, &format!("{}.bias_ih_l0_reverse", prefix))?;
        let bias_hh_rev = loader.load_component_parameter(component, &format!("{}.bias_hh_l0_reverse", prefix))?;
        
        // Load reverse gates
        self.reverse_gates.load_from_stacked_weights(
            &weight_ih_rev, &weight_hh_rev, &bias_ih_rev, &bias_hh_rev, self.hidden_size
        )?;
        
        println!("✅ TextEncoderLSTM loaded: 512→512 bidirectional with PyTorch stacked weights");
        Ok(())
    }
}

/// Load weights for ProsodyLSTM from Kokoro's predictor component
impl LoadWeightsBinary for ProsodyLSTM {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        println!("Loading ProsodyLSTM weights for {}.{}", component, prefix);
        
        // STRICT: Load forward direction weights
        let weight_ih_fwd = loader.load_component_parameter(component, &format!("{}.weight_ih_l0", prefix))?;
        let weight_hh_fwd = loader.load_component_parameter(component, &format!("{}.weight_hh_l0", prefix))?;
        let bias_ih_fwd = loader.load_component_parameter(component, &format!("{}.bias_ih_l0", prefix))?;
        let bias_hh_fwd = loader.load_component_parameter(component, &format!("{}.bias_hh_l0", prefix))?;
        
        // STRICT: Validate shapes for Prosody (640 input features = 512 + 128)
        assert_eq!(weight_ih_fwd.shape(), &[1024, 640],
            "STRICT: Prosody weight_ih_l0 must have shape [1024, 640], got {:?}", weight_ih_fwd.shape());
        assert_eq!(weight_hh_fwd.shape(), &[1024, 256],
            "STRICT: Prosody weight_hh_l0 must have shape [1024, 256], got {:?}", weight_hh_fwd.shape());
            
        // Load forward gates
        self.forward_gates.load_from_stacked_weights(
            &weight_ih_fwd, &weight_hh_fwd, &bias_ih_fwd, &bias_hh_fwd, self.hidden_size
        )?;
        
        // STRICT: Load reverse direction weights
        let weight_ih_rev = loader.load_component_parameter(component, &format!("{}.weight_ih_l0_reverse", prefix))?;
        let weight_hh_rev = loader.load_component_parameter(component, &format!("{}.weight_hh_l0_reverse", prefix))?;
        let bias_ih_rev = loader.load_component_parameter(component, &format!("{}.bias_ih_l0_reverse", prefix))?;
        let bias_hh_rev = loader.load_component_parameter(component, &format!("{}.bias_hh_l0_reverse", prefix))?;
        
        // Load reverse gates
        self.reverse_gates.load_from_stacked_weights(
            &weight_ih_rev, &weight_hh_rev, &bias_ih_rev, &bias_hh_rev, self.hidden_size
        )?;
        
        println!("✅ ProsodyLSTM loaded: 640→512 bidirectional with style conditioning support");
        Ok(())
    }
}

/// DurationEncoderLSTM: 640→512 for multi-layer duration encoding with normalization
pub type DurationEncoderLSTM = ProsodyLSTM;

/// Helper functions
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_textencoder_lstm_shapes() {
        let lstm = TextEncoderLSTM::new(512, 256);
        let batch_size = 2;
        let seq_length = 10;
        
        let input = Tensor::from_data(
            vec![0.1; batch_size * seq_length * 512], 
            vec![batch_size, seq_length, 512]
        );
        let input_lengths = vec![8, 10]; // Variable lengths
        
        let (output, _) = lstm.forward_batch_first_with_lengths(&input, &input_lengths, None, None);
        
        // Bidirectional output should be 512 features (256×2)
        assert_eq!(output.shape(), &[batch_size, seq_length, 512]);
    }
    
    #[test]
    fn test_prosody_lstm_shapes() {
        let lstm = ProsodyLSTM::new(640, 256);
        let batch_size = 1;
        let seq_length = 5;
        
        let input = Tensor::from_data(
            vec![0.2; batch_size * seq_length * 640],
            vec![batch_size, seq_length, 640]
        );
        let input_lengths = vec![5];
        
        let (output, _) = lstm.forward_batch_first_with_lengths(&input, &input_lengths, None, None);
        
        // Bidirectional output should be 512 features (256×2)  
        assert_eq!(output.shape(), &[batch_size, seq_length, 512]);
    }
    
    #[test]
    fn test_gate_weight_extraction() {
        // Test PyTorch stacked weight parsing
        let hidden_size = 256;
        let input_size = 512;
        
        // Create a stacked weight tensor [1024, 512] = [4*256, 512]
        let mut stacked_data = vec![0.0; 1024 * 512];
        
        // Fill with pattern to test extraction
        for gate in 0..4 {
            for h in 0..hidden_size {
                for i in 0..input_size {
                    let row = gate * hidden_size + h;
                    let value = (gate as f32) * 100.0 + (h as f32) * 0.1 + (i as f32) * 0.001;
                    stacked_data[row * input_size + i] = value;
                }
            }
        }
        
        let stacked_weight = Tensor::from_data(stacked_data, vec![1024, 512]);
        
        // Extract input gate weights
        let input_gate = extract_gate_weights(&stacked_weight, LSTMGate::Input, hidden_size);
        assert_eq!(input_gate.shape(), &[hidden_size, input_size]);
        
        // Verify extraction correctness
        assert!((input_gate[&[0, 0]] - 0.0).abs() < 1e-6); // gate=0, h=0, i=0
        assert!((input_gate[&[1, 0]] - 0.1).abs() < 1e-6); // gate=0, h=1, i=0
        
        // Extract forget gate weights  
        let forget_gate = extract_gate_weights(&stacked_weight, LSTMGate::Forget, hidden_size);
        assert!((forget_gate[&[0, 0]] - 100.0).abs() < 1e-6); // gate=1, h=0, i=0
    }
}