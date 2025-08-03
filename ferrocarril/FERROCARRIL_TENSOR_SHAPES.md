# Ferrocarril Tensor Shape Validation Framework

**CRITICAL PRINCIPLE: Zero Tolerance for Silent Tensor Adaptations**
- Every tensor operation must match PyTorch behavior exactly
- No automatic reshaping or dimension inference
- Explicit assertions for all shape transformations
- Fail fast with clear error messages on dimension mismatches

## PyTorch Hidden Behaviors That Must Be Explicit in Rust

### 1. Automatic Broadcasting Rules

**PyTorch Implicit:**
```python
# Automatic broadcasting: [B, 1, T] mask to [B, C, T] tensor
x.masked_fill_(mask, 0.0)  # PyTorch handles broadcast automatically
```

**Rust Explicit Required:**
```rust
// Must validate dimensions explicitly before broadcasting
assert_eq!(mask.shape()[0], x.shape()[0], "Batch dimensions must match");
assert_eq!(mask.shape()[2], x.shape()[2], "Time dimensions must match");
assert_eq!(mask.shape()[1], 1, "Mask channel dimension must be 1 for broadcasting");

// Manual broadcasting implementation required
for b in 0..batch_size {
    for c in 0..channels {
        for t in 0..time {
            if mask[&[b, 0, t]] {  // Explicit 0 for the 1-dimension
                x[&[b, c, t]] = 0.0;
            }
        }
    }
}
```

### 2. Dynamic Tensor Reshaping

**PyTorch Implicit:**
```python
# Automatic shape inference with -1
h = h.view(h.size(0), h.size(1), 1)  # [B, 2*C] → [B, 2*C, 1]
gamma, beta = torch.chunk(h, chunks=2, dim=1)  # Split along dim 1
```

**Rust Explicit Required:**
```rust
// Must calculate target shapes explicitly
let h_shape = h.shape();
let target_shape = vec![h_shape[0], h_shape[1], 1];
let h_reshaped = h.reshape(&target_shape);

// Explicit chunking with dimension validation
assert_eq!(h_reshaped.shape()[1] % 2, 0, "Channel dimension must be even for chunking");
let chunk_size = h_reshaped.shape()[1] / 2;
let gamma = h_reshaped.slice_range(&[.., 0..chunk_size, ..]);
let beta = h_reshaped.slice_range(&[.., chunk_size..(chunk_size*2), ..]);
```

### 3. Advanced Indexing Operations

**PyTorch Implicit:**
```python
# Complex indexing with two index tensors
pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
```

**Rust Explicit Required:**
```rust
// Must validate index bounds and implement manually
assert!(indices.data().iter().all(|&i| i < pred_aln_trg.shape()[0] as i64),
        "All indices must be within bounds");

for (pos, &idx) in indices.data().iter().enumerate() {
    assert!(pos < pred_aln_trg.shape()[1], "Position index out of bounds");
    pred_aln_trg[&[idx as usize, pos]] = 1.0;
}
```

### 4. Transpose and Permutation Chains

**PyTorch Implicit:**
```python
# Chain of transposes that PyTorch optimizes
x = x.transpose(-1, -2).transpose(1, -1)  # Multiple transposes
```

**Rust Explicit Required:**
```rust
// Each transpose must be explicit with shape validation
let x_t1 = transpose(&x, &[-1, -2]);  // Swap last two dims
assert_eq!(x_t1.shape(), expected_shape_after_t1);

let x_t2 = transpose(&x_t1, &[1, -1]);  // Swap dims 1 and last  
assert_eq!(x_t2.shape(), expected_final_shape);
```

## Component-Specific Tensor Flow Validation

### TextEncoder Strict Validation

```rust
impl TextEncoder {
    pub fn forward(&self, x: &Tensor<i64>, lengths: &[usize], mask: &Tensor<bool>) -> Tensor<f32> {
        // STRICT INPUT VALIDATION
        assert_eq!(x.shape().len(), 3, "Input must be [B, T] but got: {:?}", x.shape());
        assert_eq!(mask.shape(), x.shape(), "Mask shape must match input: mask={:?}, input={:?}", 
                   mask.shape(), x.shape());
        
        // 1. Embedding: [B, T] → [B, T, C]
        let embedded = self.embedding.forward(x);  // [B, T, channels]
        assert_eq!(embedded.shape(), &[x.shape()[0], x.shape()[1], self.channels],
                   "Embedding output shape mismatch");
        
        // 2. Transpose: [B, T, C] → [B, C, T] 
        let x_bct = self.transpose_btc_to_bct(&embedded);
        assert_eq!(x_bct.shape(), &[x.shape()[0], self.channels, x.shape()[1]],
                   "Transpose shape mismatch");
        
        // 3. CNN processing with mask validation
        let mut x_cnn = x_bct;
        for (i, cnn_block) in self.cnn.iter().enumerate() {
            x_cnn = cnn_block.forward(&x_cnn);
            
            // CRITICAL: Verify output shape hasn't changed unexpectedly
            assert_eq!(x_cnn.shape(), x_bct.shape(),
                       "CNN block {} changed tensor shape unexpectedly: {:?} → {:?}",
                       i, x_bct.shape(), x_cnn.shape());
            
            // Apply mask - must validate broadcasting compatibility
            self.apply_mask_with_validation(&mut x_cnn, mask);
        }
        
        // 4. Back to [B, T, C] for LSTM
        let x_btc = self.transpose_bct_to_btc(&x_cnn);
        
        // 5. Bidirectional LSTM - CRITICAL VALIDATION
        let (fw_out, _) = self.lstm_fw.forward_batch_first(&x_btc, None, None);
        let x_rev = self.reverse_time_dimension(&x_btc);
        let (bw_out_rev, _) = self.lstm_bw.forward_batch_first(&x_rev, None, None);
        let bw_out = self.reverse_time_dimension(&bw_out_rev);
        
        // STRICT CONCATENATION VALIDATION
        assert_eq!(fw_out.shape(), bw_out.shape(),
                   "Forward and backward LSTM outputs must have same shape: fw={:?}, bw={:?}",
                   fw_out.shape(), bw_out.shape());
        
        let concatenated = self.concat_bidirectional_strict(&fw_out, &bw_out);
        
        // 6. Final transpose back to [B, C, T]
        let final_output = self.transpose_btc_to_bct(&concatenated);
        
        // FINAL OUTPUT VALIDATION
        let expected_shape = &[x.shape()[0], self.channels, x.shape()[1]];
        assert_eq!(final_output.shape(), expected_shape,
                   "Final output shape mismatch: got {:?}, expected {:?}",
                   final_output.shape(), expected_shape);
        
        final_output
    }
    
    /// Apply mask with strict broadcasting validation
    fn apply_mask_with_validation(&self, tensor: &mut Tensor<f32>, mask: &Tensor<bool>) {
        // STRICT: No silent broadcasting - validate dimensions exactly
        assert_eq!(tensor.shape()[0], mask.shape()[0], "Batch dimension mismatch");
        assert_eq!(tensor.shape()[2], mask.shape()[1], 
                   "Time dimension mismatch: tensor time={}, mask time={}", 
                   tensor.shape()[2], mask.shape()[1]);
        
        // Manual broadcasting - explicit for every element
        for b in 0..mask.shape()[0] {
            for t in 0..mask.shape()[1] {
                if mask[&[b, t]] {
                    for c in 0..tensor.shape()[1] {
                        tensor[&[b, c, t]] = 0.0;
                    }
                }
            }
        }
    }
    
    /// Strict bidirectional concatenation - no shape assumptions
    fn concat_bidirectional_strict(&self, fw: &Tensor<f32>, bw: &Tensor<f32>) -> Tensor<f32> {
        // PyTorch: torch.cat([fw, bw], dim=-1)  # Concatenate along last dimension
        
        // STRICT VALIDATION: Exactly match PyTorch behavior
        assert_eq!(fw.shape()[..2], bw.shape()[..2], 
                   "Forward and backward tensor first 2 dimensions must match");
        assert_eq!(fw.shape()[2], bw.shape()[2],
                   "Forward and backward hidden dimensions must match");
        
        let (batch, seq_len, hidden_size) = (fw.shape()[0], fw.shape()[1], fw.shape()[2]);
        let final_hidden = hidden_size * 2;  // Bidirectional doubles hidden size
        
        let mut result = vec![0.0; batch * seq_len * final_hidden];
        
        for b in 0..batch {
            for t in 0..seq_len {
                // Copy forward output to first half
                for h in 0..hidden_size {
                    result[b * seq_len * final_hidden + t * final_hidden + h] = fw[&[b, t, h]];
                }
                // Copy backward output to second half
                for h in 0..hidden_size {
                    result[b * seq_len * final_hidden + t * final_hidden + hidden_size + h] = bw[&[b, t, h]];
                }
            }
        }
        
        Tensor::from_data(result, vec![batch, seq_len, final_hidden])
    }
}
```

## Strict Validation Framework for All Components

### Input Validation Template

```rust
macro_rules! validate_tensor_shape {
    ($tensor:expr, $expected:expr, $context:expr) => {
        assert_eq!($tensor.shape(), $expected,
                   "{}: Expected shape {:?}, got {:?}", 
                   $context, $expected, $tensor.shape());
    };
}

macro_rules! validate_tensor_compatible {
    ($tensor1:expr, $tensor2:expr, $dims:expr, $context:expr) => {
        for &dim in $dims {
            assert_eq!($tensor1.shape()[dim], $tensor2.shape()[dim],
                       "{}: Dimension {} mismatch: {} vs {}",
                       $context, dim, $tensor1.shape()[dim], $tensor2.shape()[dim]);
        }
    };
}
```

### Component Validation Rules

#### 1. BERT/Albert Component
```rust
// PyTorch: bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
pub fn forward(&self, input_ids: &Tensor<i64>, attention_mask: &Tensor<bool>) -> Tensor<f32> {
    validate_tensor_shape!(input_ids, &[batch_size, seq_length], "BERT input_ids");
    validate_tensor_shape!(attention_mask, &[batch_size, seq_length], "BERT attention_mask");
    
    // NO silent shape adaptation allowed
    assert!(input_ids.shape()[1] <= self.max_position_embeddings,
            "Sequence length {} exceeds max position embeddings {}",
            input_ids.shape()[1], self.max_position_embeddings);
    
    // Process with strict shape tracking at each layer...
}
```

#### 2. ProsodyPredictor Component
```rust
// PyTorch: duration = torch.sigmoid(duration).sum(axis=-1) / speed
pub fn predict_duration(&self, x: &Tensor<f32>, speed: f32) -> Tensor<f32> {
    let dur_logits = self.duration_proj.forward(x);
    validate_tensor_shape!(dur_logits, &[batch_size, seq_len, self.max_dur], "Duration logits");
    
    // Explicit sigmoid and sum - no automatic reduction
    let dur_sigmoid = dur_logits.sigmoid();
    let dur_summed = dur_sigmoid.sum_over_dim(-1);  // Explicit dimension specification
    
    assert_eq!(dur_summed.shape(), &[batch_size, seq_len],
               "Duration sum shape mismatch");
    
    let durations = dur_summed.div_scalar(speed);
    durations.round().clamp_min(1.0)  // Explicit clamp bounds
}
```

#### 3. Alignment Matrix Creation
```rust
// PyTorch: pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
pub fn create_alignment_matrix(&self, durations: &Tensor<f32>) -> Tensor<f32> {
    let seq_len = durations.shape()[1];
    let total_frames: usize = durations.data().iter().map(|&d| d.round() as usize).sum();
    
    // Create indices with strict bounds checking
    let mut indices = vec![0usize; total_frames];
    let mut frame_idx = 0;
    
    for token_idx in 0..seq_len {
        let dur = durations[&[0, token_idx]].round() as usize;
        assert!(dur >= 1, "Duration must be at least 1 frame");
        
        for _ in 0..dur {
            assert!(frame_idx < total_frames, "Frame index overflow");
            indices[frame_idx] = token_idx;
            frame_idx += 1;
        }
    }
    
    // Create alignment matrix with explicit bounds checking
    let mut alignment = Tensor::zeros(&[seq_len, total_frames]);
    for (frame, &token) in indices.iter().enumerate() {
        assert!(token < seq_len, "Token index {} out of bounds", token);
        assert!(frame < total_frames, "Frame index {} out of bounds", frame);
        alignment[&[token, frame]] = 1.0;
    }
    
    alignment.unsqueeze(0)  // Add batch dimension explicitly
}
```

## Critical Assertion Framework

### Shape Tracking at Every Step

```rust
pub struct TensorFlowTracker {
    step: usize,
    expected_shapes: HashMap<String, Vec<usize>>,
}

impl TensorFlowTracker {
    pub fn validate_step(&mut self, name: &str, tensor: &Tensor<f32>, expected: &[usize]) {
        self.step += 1;
        println!("Step {}: Validating tensor '{}' shape", self.step, name);
        
        assert_eq!(tensor.shape(), expected,
                   "Step {}: Tensor '{}' shape mismatch. Expected {:?}, got {:?}",
                   self.step, name, expected, tensor.shape());
        
        self.expected_shapes.insert(name.to_string(), expected.to_vec());
        println!("✅ Step {}: {} shape validated: {:?}", self.step, name, expected);
    }
    
    pub fn validate_transformation(&self, from: &str, to: &str, tensor: &Tensor<f32>) {
        if let Some(from_shape) = self.expected_shapes.get(from) {
            println!("Validating transformation: {} → {}", from, to);
            println!("  Input shape: {:?}", from_shape);
            println!("  Output shape: {:?}", tensor.shape());
            
            // Validate that transformation makes sense
            let from_elements: usize = from_shape.iter().product();
            let to_elements: usize = tensor.shape().iter().product();
            
            // For reshapes, element count must be preserved
            if to.contains("reshape") || to.contains("view") {
                assert_eq!(from_elements, to_elements,
                           "Reshape must preserve element count: {} → {}",
                           from_elements, to_elements);
            }
        }
    }
}
```

## TextEncoder Exact PyTorch Matching

Based on the analysis, here's what the TextEncoder must implement exactly:

### Critical Shape Transformations

1. **Embedding**: `[B, T] → [B, T, channels]`
2. **Initial Transpose**: `[B, T, channels] → [B, channels, T]`
3. **CNN Processing**: `[B, channels, T] → [B, channels, T]` (preserve shape)
4. **LSTM Transpose**: `[B, channels, T] → [B, T, channels]`
5. **Bidirectional LSTM**: `[B, T, channels] → [B, T, channels*2]`
6. **Final Transpose**: `[B, T, channels*2] → [B, channels*2, T]`

### Mask Broadcasting Rules

- Input mask: `[B, T]`
- For CNN layers: Broadcast to `[B, channels, T]` 
- For LSTM: Use as `[B, T]` directly
- **NO AUTOMATIC ADAPTATION**

### LSTM Pack/Unpack Behavior

PyTorch uses `pack_padded_sequence` which Rust must implement explicitly:

```rust
// PyTorch: x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
pub fn pack_padded_sequence(&self, x: &Tensor<f32>, lengths: &[usize]) -> PackedSequence {
    // Validate lengths are within bounds
    for (batch_idx, &length) in lengths.iter().enumerate() {
        assert!(length <= x.shape()[1], 
                "Length {} for batch {} exceeds sequence length {}",
                length, batch_idx, x.shape()[1]);
        assert!(length > 0, "Length must be positive for batch {}", batch_idx);
    }
    
    // Implementation must match PyTorch's packed format exactly
    // This operation is complex and critical for proper LSTM behavior
}
```

## Zero-Tolerance Validation Rules

1. **No Silent Dimension Mismatches**: Every operation validates input shapes
2. **No Implicit Broadcasting**: All broadcasting must be explicit and validated  
3. **No Shape Inference**: All target shapes must be calculated explicitly
4. **No Fallback Values**: Any unexpected input must cause immediate failure
5. **No Device Assumptions**: All tensor device compatibility must be explicit

This framework ensures that every single tensor operation in Rust matches the exact behavior of the PyTorch reference implementation, with no hidden surprises or silent adaptations.