//! LSTM implementation optimized for CPU execution

use crate::{Parameter, Forward};
use ferrocarril_core::tensor::Tensor;

#[derive(Debug)]
pub struct LSTM {
    // Weights for input-to-hidden connections
    weight_ih: Parameter,
    // Weights for hidden-to-hidden connections
    weight_hh: Parameter,
    // Bias for input-to-hidden
    bias_ih: Option<Parameter>,
    // Bias for hidden-to-hidden
    bias_hh: Option<Parameter>,
    
    // Configuration
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
    batch_first: bool,
}

impl LSTM {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        batch_first: bool,
        bidirectional: bool,
    ) -> Self {
        // For now we only implement single layer
        assert_eq!(num_layers, 1, "Only single layer LSTM supported for MVP");
        
        // Gates are ordered: input, forget, cell, output
        let gate_size = 4 * hidden_size;
        
        // Initialize weights
        let weight_ih = Parameter::new(Tensor::new(vec![gate_size, input_size]));
        let weight_hh = Parameter::new(Tensor::new(vec![gate_size, hidden_size]));
        
        let bias_ih = Some(Parameter::new(Tensor::new(vec![gate_size])));
        let bias_hh = Some(Parameter::new(Tensor::new(vec![gate_size]))); 
        
        Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            batch_first,
        }
    }
    
    /// Compute LSTM gates for a single timestep
    fn compute_gates(&self, input: &[f32], hidden: &[f32]) -> Vec<f32> {
        let mut gates = vec![0.0; 4 * self.hidden_size];
        
        // Compute input contribution
        let weight_ih = self.weight_ih.data();
        for i in 0..4 * self.hidden_size {
            let mut sum = 0.0;
            for j in 0..self.input_size {
                sum += weight_ih[&[i, j]] * input[j];
            }
            gates[i] = sum;
        }
        
        // Compute hidden contribution
        let weight_hh = self.weight_hh.data();
        for i in 0..4 * self.hidden_size {
            let mut sum = 0.0;
            for j in 0..self.hidden_size {
                sum += weight_hh[&[i, j]] * hidden[j];
            }
            gates[i] += sum;
        }
        
        // Add biases
        if let Some(ref bias_ih) = self.bias_ih {
            let bias_ih_data = bias_ih.data();
            for i in 0..4 * self.hidden_size {
                gates[i] += bias_ih_data[&[i]];
            }
        }
        
        if let Some(ref bias_hh) = self.bias_hh {
            let bias_hh_data = bias_hh.data();
            for i in 0..4 * self.hidden_size {
                gates[i] += bias_hh_data[&[i]];
            }
        }
        
        gates
    }
    
    /// Apply activation functions to gates and compute new hidden and cell states
    fn apply_activations(
        &self,
        gates: &[f32],
        cell: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let hidden_size = self.hidden_size;
        
        // Extract gates
        let input_gate = &gates[0..hidden_size];
        let forget_gate = &gates[hidden_size..2 * hidden_size];
        let cell_gate = &gates[2 * hidden_size..3 * hidden_size];
        let output_gate = &gates[3 * hidden_size..4 * hidden_size];
        
        let mut new_cell = vec![0.0; hidden_size];
        let mut new_hidden = vec![0.0; hidden_size];
        
        // Apply activations and compute new states
        for i in 0..hidden_size {
            let ig = sigmoid(input_gate[i]);
            let fg = sigmoid(forget_gate[i]);
            let cg = tanh(cell_gate[i]);
            let og = sigmoid(output_gate[i]);
            
            new_cell[i] = fg * cell[i] + ig * cg;
            new_hidden[i] = og * tanh(new_cell[i]);
        }
        
        (new_hidden, new_cell)
    }
    
    /// Forward pass for the LSTM
    pub fn forward_unbatched(
        &self,
        input: &Tensor<f32>,
        hidden: Option<&Tensor<f32>>,
        cell: Option<&Tensor<f32>>,
    ) -> (Tensor<f32>, (Tensor<f32>, Tensor<f32>)) {
        let shape = input.shape();
        let (seq_len, input_size) = (shape[0], shape[1]);
        assert_eq!(input_size, self.input_size);
        
        // Initialize states if not provided
        let initial_hidden = match hidden {
            Some(h) => h.data().to_vec(),
            None => vec![0.0; self.hidden_size],
        };
        let initial_cell = match cell {
            Some(c) => c.data().to_vec(), 
            None => vec![0.0; self.hidden_size],
        };
        
        let mut hidden_state = initial_hidden;
        let mut cell_state = initial_cell;
        let mut outputs = Vec::new();
        
        // Process each timestep
        for t in 0..seq_len {
            // Extract input at timestep t
            let mut input_t = vec![0.0; input_size];
            for i in 0..input_size {
                input_t[i] = input[&[t, i]];
            }
            
            // Compute gates and new states
            let gates = self.compute_gates(&input_t, &hidden_state);
            let (new_hidden, new_cell) = self.apply_activations(&gates, &cell_state);
            
            hidden_state = new_hidden;
            cell_state = new_cell;
            outputs.extend_from_slice(&hidden_state);
        }
        
        // Create output tensors
        let output = Tensor::from_data(outputs, vec![seq_len, self.hidden_size]);
        let final_hidden = Tensor::from_data(hidden_state, vec![self.hidden_size]);
        let final_cell = Tensor::from_data(cell_state, vec![self.hidden_size]);
        
        (output, (final_hidden, final_cell))
    }
}

impl Forward for LSTM {
    type Output = (Tensor<f32>, (Tensor<f32>, Tensor<f32>));
    
    fn forward(&self, input: &Tensor<f32>) -> Self::Output {
        // For MVP, only support unbatched forward
        if self.batch_first {
            // TODO: Implement batched forward
            panic!("Batched LSTM not implemented yet");
        } else {
            self.forward_unbatched(input, None, None)
        }
    }
}

/// Sigmoid activation function
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Tanh activation function
fn tanh(x: f32) -> f32 {
    x.tanh()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lstm_forward_unbatched() {
        let lstm = LSTM::new(10, 20, 1, false, false);
        let input = Tensor::new(vec![5, 10]); // [seq_len, input_size]
        let (output, (hidden, cell)) = lstm.forward(&input);
        
        assert_eq!(output.shape(), &[5, 20]);
        assert_eq!(hidden.shape(), &[20]);
        assert_eq!(cell.shape(), &[20]);
    }
}