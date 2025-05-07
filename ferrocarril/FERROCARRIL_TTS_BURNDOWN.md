# Ferrocarril TTS Implementation Burndown List

> **Critical Implementation Principles**
>
> 1. **Zero Tolerance for Silent Error Masking:** Never implement fallbacks that hide real errors.
> 2. **No Fake Weights:** Never use random initialization when weights should be loaded.
> 3. **No Dimensional Fudging:** Never silently reshape or project tensors when dimensions don't match.
> 4. **Honor the Reference:** Always match the exact shape transformations of the Kokoro implementation.
> 5. **Fail Fast:** Panic immediately with clear error messages when deviating from the reference.
> 6. **Functional Correctness Over Structural Matching:** Verify components transform data meaningfully, not just match shapes. All-zero outputs from non-zero inputs are a critical red flag.

This document serves as the critical burndown list for the Ferrocarril TTS system implementation based on systematic analysis of tensor shape mismatches and other implementation discrepancies compared to the reference Kokoro implementation.

## Testing Principles

### Verifying Functional Correctness

Component tests must go beyond shape verification and validate meaningful data transformation:

- [ ] **Load Real Weights**: Every component test MUST load actual weights from the model file to verify functional behavior, not just structure
```rust
// Required pattern:
let loader = BinaryWeightLoader::from_directory("ferrocarril_weights")?;
component.load_weights_binary(&loader, "component_name", "prefix")?;

// Then test with real data
let output = component.forward(&input);

// All-zero outputs are a red flag - verify data transformation
assert!(!output.data().iter().all(|&v| v.abs() < 1e-6),
    "Component produces zero output with real weights - functionally dead!");
```

- [ ] **Validate Data Transformation**: Check that outputs have expected statistical properties for given inputs
```rust
// Basic validation - outputs are meaningfully different from inputs
let mean: f32 = output.data().iter().sum::<f32>() / output.data().len() as f32;
let variance = calculate_variance(output.data(), mean);

// Outputs should have reasonable statistics for a neural network component
assert!(variance > min_expected_variance, 
    "Output lacks statistical variation - functionally suspicious!");
```

- [ ] **End-to-End Validation**: Test full pipeline with known inputs and expected outputs
```rust
// Validate against reference implementation outputs where possible,
// or at minimum verify that audio output contains non-zero, non-constant sample values
```

## High-Priority Issues

### 1. LSTM Implementation (CRITICAL)

The current LSTM implementation has fundamental flaws that affect all sequence processing:

- [ ] **Fix Fake Bidirectionality**: Despite claiming bidirectional support, the implementation only processes in a single direction
```rust
// Current issue:
// Bidirectional LSTMs initialized but only forward direction is used

// Required implementation:
// 1. Process sequence in forward direction using forward weights
// 2. Process sequence in reverse direction using reverse weights 
// 3. Concatenate outputs along the feature dimension
```

- [ ] **Use Reverse Weights**: The implementation ignores `*_reverse` weights which contain half of the model's capacity
```rust
// Expected weight loading:
if self.bidirectional {
    // Load weights for forward direction
    *self.weight_ih_l0 = Parameter::new(loader.load_tensor(&format!("{}.weight_ih_l0", prefix))?);
    *self.weight_hh_l0 = Parameter::new(loader.load_tensor(&format!("{}.weight_hh_l0", prefix))?);
    *self.bias_ih_l0 = Parameter::new(loader.load_tensor(&format!("{}.bias_ih_l0", prefix))?);
    *self.bias_hh_l0 = Parameter::new(loader.load_tensor(&format!("{}.bias_hh_l0", prefix))?);

    // Load weights for reverse direction (currently missing)
    *self.weight_ih_l0_reverse = Parameter::new(loader.load_tensor(&format!("{}.weight_ih_l0_reverse", prefix))?);
    *self.weight_hh_l0_reverse = Parameter::new(loader.load_tensor(&format!("{}.weight_hh_l0_reverse", prefix))?);
    *self.bias_ih_l0_reverse = Parameter::new(loader.load_tensor(&format!("{}.bias_ih_l0_reverse", prefix))?);
    *self.bias_hh_l0_reverse = Parameter::new(loader.load_tensor(&format!("{}.bias_hh_l0_reverse", prefix))?);
}
```

- [ ] **Implement True Variable-Length Sequence Support**: Either implement pack_padded_sequence, or ensure hidden states aren't updated past sequence end
```rust
// Current issue: runs LSTM over padding tokens
// Fix options:
// 1. Implement pack/unpack:
// let packed_sequence = pack_padded_sequence(x, lengths);
// let (output, _) = self.lstm(packed_sequence);
// let unpacked_output = pad_packed_sequence(output);

// 2. Or at minimum, freeze hidden state at sequence end:
for t in 0..max_length {
    for b in 0..batch_size {
        if t < lengths[b] {  // Only update if within sequence length
            // Update hidden state with lstm cell
        }
    }
}
```

- [ ] **Remove Silent Projection Fallback**: LSTM silently reshapes inputs when dimensions don't match, hiding real errors
```rust
// Current issue:
if feat != self.input_size {
    println!("WARNING: Input feature dimension {} does not match LSTM input_size {}.", 
             feat, self.input_size);
    println!("This is likely due to a configuration mismatch. Using a projection to match dimensions.");
    
    // Create projected version - REMOVE THIS, LET IT FAIL INSTEAD
    ...
}

// Fix: Remove the silent reshaping and use assert! instead:
assert_eq!(feat, self.input_size, 
    "Input feature dimension {} does not match LSTM input_size {}", feat, self.input_size);
```

### 2. ProsodyPredictor Implementation (CRITICAL)

The ProsodyPredictor implementation has severe issues affecting audio quality:

- [ ] **Fix Style Application**: Style is applied inconsistently, sometimes applied twice
```rust
// Current issue: style concatenated in DurationEncoder and then again in forward

// Fix:
// 1. If style is already included in DurationEncoder's output,
//    don't concatenate it again in forward()
// 2. Or remove style from DurationEncoder and only apply in forward()
```

- [ ] **Fix Energy Pooling Dimension**: The `en` tensor is missing style dimensions
```rust
// Current issue:
// Kokoro: en shape is [B, H+style_dim, S]
// Ferrocarril: en shape is [B, H, S]

// Fix: Ensure d_enc includes style dimension in pooling
// en = d_enc.transpose(-1, -2) @ pred_aln_trg  // Here d_enc must have shape [B, T, H+style_dim]
```

- [ ] **Fix LSTM Dimension Order**: The LSTM in Kokoro receives inputs in a specific order
```python
# In Kokoro's predict_f0_noise:
x, _ = self.shared(x.transpose(-1, -2))  # LSTM expects [B, T, C] but gets [B, C, T]
F0 = x.transpose(-1, -2)  # Back to [B, C, T]

# In Ferrocarril, we need to match this exact pattern:
x_transposed = transpose_bct_to_btc(x);  // [B, C, T] -> [B, T, C]
(x_out, _) = self.shared_lstm.forward_batch_first(&x_transposed, None, None);
f0 = transpose_btc_to_bct(&x_out);  // [B, T, C] -> [B, C, T]
```

- [ ] **Fix Dimension Swapping**: Fix critical tensor dimension wrongly transposed before F0/noise prediction
```rust
// Current issue:
en_btf = transpose(&en, &[0, 2, 1])  // This swaps channels and frames!

// Fix:
// Don't transpose en from [B, H, F] to [B, F, H]
// Simply call predict_f0_noise directly with the correct format
```

### 3. AdaIN Implementation (HIGH)

- [ ] **Fix Silent Channel Mismatch**: AdaIN silently tolerates channel mismatches, hiding errors
```rust
// Current issue:
if c != self.num_features {
    println!("Warning: Channel count mismatch in AdaIN1d: layer expects {}, input has {}. Using input channels.", 
            self.num_features, c);
    // Then continues using clipped feature map
}

// Fix:
assert_eq!(c, self.num_features, 
    "Channel count mismatch in AdaIN1d: layer expects {}, input has {}", 
    self.num_features, c);
```

- [ ] **Match Affine Parameters**: Set `affine=false` in InstanceNorm1d to match Kokoro
```rust
// Current issue:
// Kokoro: nn.InstanceNorm1d(C, affine=False, eps=1e-5)
// Ferrocarril: InstanceNorm1d::new(C, 1e-5, affine=true)

// Fix:
self.norm = InstanceNorm1d::new(num_features, 1e-5, false);
```

- [ ] **Fix γ/β Indexing**: Fix edge cases for slicing style parameters
```rust
// Current issue:
if fc.out_features < 2 * C, duplicates β indices

// Fix:
assert_eq!(fc.out_features, 2 * self.num_features, 
    "fc output must have twice the channels (got {}, expected {})", 
    fc.out_features, 2 * self.num_features);
```

### 4. BERT Implementation (HIGH)

- [ ] **Fix Embedding Size Configuration**: Remove hard-coded embedding size
```rust
// Current issue:
let embedding_size = 128;  // Hard-coded

// Fix: 
let embedding_size = config.embedding_size;  // From config JSON
```

- [ ] **Implement Complete FFN**: Add the missing second linear projection
```rust
// Current issue:
Missing ffn_output Linear layer in FeedForward

// Fix:
pub struct FeedForward {
    ffn: Linear,               // hidden→intermediate
    ffn_output: Linear,        // intermediate→hidden
    ...
}

// Forward pass
let intermediate = self.ffn.forward(input);
let intermediate_act = intermediate.map(|x| gelu(x));
let output = self.ffn_output.forward(&intermediate_act);
```

- [ ] **Fix Residual Connections**: Add proper pre-norm style residual connections
```rust
// Current issue:
// residual connection skips attention and only adds feed-forward output

// Fix: proper pre-LayerNorm residual connection pattern:
let attn_out = self.attention.forward(hidden_states, attention_mask);
let hidden_after_attn = hidden_states + attn_out;  // First residual connection
let feed_forward_out = self.feed_forward.forward(&hidden_after_attn);
let output = hidden_after_attn + feed_forward_out;  // Second residual connection
self.full_layer_norm.forward(&output)
```

- [ ] **Fix Attention Mask Convention**: Ensure mask conventions match
```rust
// Current issue:
// Rust comment says "1 indicates masked positions" 
// but HF expects 0=masked, 1=visible

// Fix:
// Ensure mask input to attention.forward() has correct semantics
// Where 0 = masked (padding) position
// And 1 = valid token position
```

### 5. Decoder (MEDIUM)

- [ ] **Fix F0 Tensor Layout**: Correct shape transformation for F0 input
```python
# In Kokoro:
f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # [B, 1, T] -> [B, T, 1]

# In Ferrocarril:
// Currently: [B, 1, T] throughout
// Fix: Transpose to proper shape [B, T, 1] for source module
```

- [x] **Fix AdainResBlk1d Upsampling**: Add proper residual path handling
```rust
// Current issue: 
// Upsampling in main path but not in residual, or vice versa

// Fix (implemented):
// Always match upsampling in both main + residual paths with 1x1 convolution if needed
// 1. Added learned shortcut flag and conv1x1 fields to AdaINResBlock1
// 2. Split forward into _shortcut and _residual methods
// 3. Applied upsampling to both paths
// 4. Combined paths with normalization by 1/sqrt(2)
```

- [ ] **Fix Reflection Padding**: Apply padding on the left only
```rust
// Current issue:
// Padding applied on wrong side

// Fix:
// Replace with left-side reflection only, or use a proper ReflectionPad1d
```

### 6. Voice Embedding Handling (MEDIUM)

- [ ] **Fix Voice File Shape Handling**: Properly convert from [510, 1, 256] to [1, style_dim * 2]
```rust
// Current issue:
// Ferrocarril loads flattened [1, 130560]
// Needs [1, 256] with first/second half for reference/style

// Fix: Select the appropriate sequence position (e.g., middle) and format
// Extract style_dim elements for each half
```

- [ ] **Fix Reference/Style Split**: Split embedding into reference and style parts
```rust
// Current issue: 
// Inconsistent extraction of reference vs style parts

// Fix:
// Consistently split into ref_embedding and style_embedding:
let ref_embedding = Tensor::from_data(ref_part_data, vec![batch_size, style_dim]);
let style_embedding = Tensor::from_data(style_part_data, vec![batch_size, style_dim]);
```

### 7. Alignment Tensor (MEDIUM)

- [ ] **Fix Alignment Creation**: Properly expand durations into alignment matrix
```rust
// Current issue:
// Creates simple identity alignment [T, T] instead of [T, sum(durations)]

// Fix:
// Properly expand predicted durations into alignment matrix:
// 1. Predict durations
// 2. Create indices by repeating position indices
// 3. Create matrix [T, sum(durations)] with 1.0 at appropriate positions
```

- [ ] **Fix Text-Audio Mapping**: Ensure proper tensor shapes throughout pipeline
```rust
// Current issue: 
// Alignment tensor shapes not consistently handled

// Fix:
// Precisely match Kokoro's shape transformations:
// 1. durations -> indices -> alignment matrix
// 2. alignment matrix shapes: [seq_len, total_frames]
// 3. proper batch dimension handling
```

### 8. TextEncoder Implementation (MEDIUM)

- [ ] **Fix Pack/Pad Mechanism**: Implement proper variable-length sequence handling
```rust
// Current issue:
// Runs LSTM over padded tokens that should be skipped

// Fix:
// Implement effective pack/unpack, or at minimum, freeze hidden
// state updates at sequence end
```

- [ ] **Remove Silent Reshaping**: Remove the projection fallback
```rust
// Current issue: Silently reshapes inputs with `if c != channels`

// Fix:
assert_eq!(c, channels, "Channel dimension mismatch in TextEncoder");
```

### 9. General Issues (HIGH)

- [ ] **Remove All Fallbacks**: Systematically identify and remove all silent fallbacks
```rust
// Target pattern:
if some_condition {
    println!("Warning: ...");
    // Followed by silent workaround
}

// Replace with:
assert!(some_condition, "Error: Expected condition not met");
```

- [ ] **Fix Dimension Assertions**: Add proper assertions for expected tensor shapes
```rust
// Add at key points:
assert_eq!(x.shape(), expected_shape, "Shape mismatch: expected {:?}, got {:?}", 
           expected_shape, x.shape());
```

- [ ] **Fix Vocoder Implementation**: Ensure proper time dimensions 
```rust
// Current issue:
// Various shape mismatches in time dimension

// Fix:
// Pad correctly, ensure proper alignment in residual connections
```

## Implementation Principles

### Implement Exact Shape Transformations

Always match the exact tensor shapes and transformations used in Kokoro. PyTorch handles many tensor operations implicitly (broadcasting, dimension alignment) that must be explicitly implemented in Rust:

```python
# In PyTorch, this works automatically:
x = torch.randn(10, 20)
y = torch.randn(20, 30)
z = x @ y  # Shapes match automatically

# In Rust:
assert_eq!(x.shape()[1], y.shape()[0], 
    "Matrix multiplication dimension mismatch: {:?} @ {:?}", x.shape(), y.shape());
```

### Never Mask Errors

Error masking leads to subtle bugs that are hard to track down. When you see a pattern like:

```rust
if dimension_mismatch {
    println!("Warning: Dimensions don't match, working around...");
    // Code that silently works around the issue
}
```

Replace it with:

```rust
assert!(!dimension_mismatch, 
    "Fatal error: Dimensions must match, got {} but expected {}", 
    actual_dimension, expected_dimension);
```

### Never Use Random Weights

When weights are missing, don't fall back to random initialization:

```rust
// Wrong approach:
match loader.load_component_parameter(component, &weight_path) {
    Ok(weight) => { /* Use weight */ },
    Err(_) => {
        println!("Warning: Weight not found, using random initialization");
        // Continue with random weights
    }
}

// Correct approach:
let weight = loader.load_component_parameter(component, &weight_path)?;
// Will propagate error up if weight not found
```

### Document Shape Transitions

For each function that transforms tensor shapes, provide clear documentation of the expected input and output shapes:

```rust
/// Forward pass for ProsodyPredictor
/// 
/// # Shape Transformations
/// - txt_feat: [B, C, T] -> BERT encoded hidden states
/// - style: [B, style_dim] -> Style embedding
/// - text_mask: [B, T] -> Text mask (true = masked)
/// - alignment: [T, S] -> Alignment matrix mapping text to audio frames
/// 
/// # Returns
/// - durations: [B, T, max_dur] -> Duration predictions
/// - en: [B, C+style_dim, S] -> Pooled hidden states for decoder
```

### Verify Functional Data Transformation

Structural shape checking is necessary but insufficient. Real validation requires:

1. **Non-Zero Outputs**: A neural network with non-zero inputs that produces all-zero outputs is functionally dead.

2. **Statistical Properties**: Outputs should have reasonable means, variances, and distributions for given inputs.

3. **Reference Comparison**: When possible, compare outputs to reference implementation.

```rust
// Example test verification:
let output = component.forward(&input);

// All-zero outputs are a critical red flag
assert!(!output.data().iter().all(|&v| v.abs() < 1e-6), 
        "Component produces all zeros - functionally dead!");
        
// Statistical sanity check
let mean = output.data().iter().sum::<f32>() / output.data().len() as f32;
let variance = compute_variance(output.data());
assert!(variance > 0.001, "Output has suspiciously low variance!");
```

## Testing Strategy

For each component, implement tests that:

1. Verify tensor shapes match the Kokoro implementation
2. Ensure error propagation works correctly
3. Validate that fallbacks have been removed
4. **Load real weights** from the model file to verify functional correctness
5. **Validate outputs** have appropriate non-zero values and statistical properties

## Conclusion

This burndown list identifies the critical issues that must be addressed to make Ferrocarril a correct implementation of the Kokoro TTS model. The most important fixes involve the LSTM and ProsodyPredictor components, which affect all downstream processing. Always remember that shape mismatches are not just cosmetic issues - they're red flags indicating fundamental implementation errors.

The underlying pattern across all issues is a tendency to silently handle errors with fallbacks rather than failing fast. This approach masks real problems and makes the system behave unpredictably. By systematically addressing each item in this list and adhering to the implementation principles, we can ensure Ferrocarril accurately reproduces the behavior of the original Kokoro model.

Remember: A structurally correct but functionally dead system is equivalent to no system at all. Always verify that your components actually transform data in meaningful ways by testing with real weights and validating outputs.