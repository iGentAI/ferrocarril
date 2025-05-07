# CustomBERT Implementation Handover Document

## 1. Overview and Importance

The CustomBERT component represents one of the most critical missing pieces in the Ferrocarril TTS system. In the Kokoro reference implementation, this component (named CustomAlbert) provides rich contextual embeddings that significantly enhance the TTS system's prosody, stress, and overall natural sound. While other components in the Ferrocarril system have been implemented, the BERT module is currently a placeholder that simply returns constant values.

This document provides the complete context and implementation plan for adding an actual BERT implementation to Ferrocarril.

## 2. Technical Context

### 2.1 CustomAlbert in Kokoro

In the Python Kokoro implementation, CustomAlbert is a thin wrapper around Hugging Face's AlbertModel:

```python
# In kokoro/modules.py
class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state
```

This wrapper simply extracts and returns the model's last hidden state. In Kokoro's architecture, this is used as follows:

```python
# In kokoro/model.py
self.bert = CustomAlbert(AlbertConfig(vocab_size=config['n_token'], **config['plbert']))
self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, config['hidden_dim'])

# Then in the forward method:
bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
```

The key purpose of BERT here is to create contextually-aware representations of each token, allowing the model to understand dependencies between tokens and produce more natural-sounding speech.

### 2.2 Current Implementation in Ferrocarril

Currently, Ferrocarril uses placeholder tensors to simulate BERT outputs:

```rust
// Create bert_dur as a placeholder - in real implementation, this would come from BERT
let bert_hidden_size = 768; // Typically 768 for ALBERT
let bert_dur = Tensor::from_data(
    vec![0.1; batch_size * seq_len * bert_hidden_size],
    vec![batch_size, seq_len, bert_hidden_size]
);

// Create bert_encoder as placeholder - we'd need Linear implementation here
let d_en_btc = Tensor::from_data(
    vec![0.1; batch_size * seq_len * hidden_dim],
    vec![batch_size, seq_len, hidden_dim]
);
```

This placeholder approach allows the rest of the system to function but lacks the contextual understanding that makes the TTS output sound natural.

## 3. Implementation Requirements

### 3.1 High-Level Architecture

The CustomBERT implementation needs to follow this structure:

1. **CustomBERT Module**: A Rust implementation of the ALBERT transformer architecture
2. **BertEncoder**: A linear projection from BERT's hidden size to the model's hidden dimension
3. **Weight Loading**: Mechanism to load ALBERT weights from the converted binary format

### 3.2 Detailed Component Specifications

#### 3.2.1 CustomBERT Structure

The ALBERT architecture is a derivative of BERT with parameter-sharing optimizations. Its key components include:

- **Embeddings**: Token, position, and token-type embeddings
- **Encoder**: Multiple self-attention layers with parameter sharing
- **Attention Mechanism**: Multi-head self-attention
- **Feed-Forward Networks**: Position-wise fully connected layers

Based on Kokoro's config, this should use:
- 768 hidden dimension
- 12 attention heads
- 12 hidden layers
- 2048 intermediate size
- A vocabulary size matching `n_token` (178 in the current configuration)

#### 3.2.2 BertEncoder

This is a simpler component - just a linear layer that projects from ALBERT's output dimension (768) to the hidden dimension used throughout the model (512 in current configuration).

#### 3.2.3 Integration Points

The CustomBERT needs to integrate at these specific points:

1. In `FerroModel::load_binary`: Loading CustomBERT and BertEncoder weights
2. In `FerroModel::infer_with_phonemes`: Replacing placeholder tensors with actual BERT processing
3. Ensuring the output shapes match what the rest of the pipeline expects

## 4. Implementation Plan

### 4.1 Step-by-Step Approach

1. **Create the core transformer components**:
   - Create `attention.rs` to implement multi-head self-attention
   - Create `feed_forward.rs` for position-wise feed-forward networks
   - Create `embeddings.rs` for token and position embeddings
   - Create `transformer.rs` for encoder blocks with layer normalization

2. **Implement the main CustomBERT module**:
   - Create `bert.rs` for the CustomBERT implementation
   - Implement the ALBERT-specific architecture with parameter sharing
   - Create a BertConfig struct to hold configuration parameters

3. **Implement weight loading**:
   - Add weight loading support to all BERT components
   - Ensure weights are properly mapped from the binary format

4. **Integrate with FerroModel**:
   - Update FerroModel to initialize and use CustomBERT
   - Replace placeholders with actual BERT processing
   - Ensure tensor shapes flow correctly through the pipeline

5. **Add tests**:
   - Create unit tests for individual components
   - Add integration tests for the full BERT pipeline
   - Compare outputs with reference implementation if possible

### 4.2 Attention Mechanism Implementation

The attention mechanism is a core part of transformers:

```rust
// Pseudo-code for multi-head attention
struct MultiHeadAttention {
    query_projection: Vec<Linear>,
    key_projection: Vec<Linear>,
    value_projection: Vec<Linear>,
    output_projection: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor, mask: &Option<Tensor>) -> Tensor {
        // Project inputs to multi-head dimensions
        // Split into multiple heads
        // Compute scaled dot-product attention for each head
        // Concatenate head outputs
        // Project to output dimension
    }
}
```

### 4.3 Complete ALBERT Layer

Each ALBERT layer consists of attention + feed-forward network:

```rust
struct AlbertLayer {
    attention: MultiHeadAttention,
    attention_layer_norm: LayerNorm,
    feed_forward: FeedForward,
    full_layer_norm: LayerNorm,
}

impl AlbertLayer {
    fn forward(&self, hidden_states: &Tensor, attention_mask: &Option<Tensor>) -> Tensor {
        // Multi-head attention with residual connection
        // Layer norm
        // Feed-forward with residual connection
        // Layer norm
    }
}
```

### 4.4 Parameter Sharing

ALBERT uses parameter sharing across layers for efficiency. Unlike BERT, where each layer has its own weights, ALBERT shares parameters:

```rust
struct AlbertLayerGroup {
    albert_layer: AlbertLayer,
}

impl AlbertLayerGroup {
    fn forward(&self, hidden_states: &Tensor, attention_mask: &Option<Tensor>, num_repeats: usize) -> Tensor {
        // Apply the same layer repeatedly
        let mut output = hidden_states.clone();
        for _ in 0..num_repeats {
            output = self.albert_layer.forward(&output, attention_mask);
        }
        output
    }
}
```

## 5. Challenges and Considerations

### 5.1 Computational Complexity

The BERT component will be the most computationally intensive part of the inference pipeline. Key challenges include:

- **Matrix Operations**: Efficient batch matrix multiplications are crucial
- **Memory Usage**: ALBERT helps reduce memorgy usage compared to BERT, but still requires significant memory
- **Optimization**: This component will be a prime target for performance optimization

### 5.2 Weight File Structure

The BERT component weights in Kokoro's model follow a specific structure:

```
bert/
  module.embeddings.word_embeddings.weight
  module.embeddings.position_embeddings.weight
  module.embeddings.token_type_embeddings.weight
  module.embeddings.LayerNorm.weight
  module.embeddings.LayerNorm.bias
  module.encoder.embedding_hidden_mapping_in.weight
  module.encoder.embedding_hidden_mapping_in.bias
  module.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight
  module.encoder.albert_layer_groups.0.albert_layers.0.attention.query.bias
  ...and many more
```

Each weight tensor must be correctly mapped to the corresponding part of the ALBERT architecture.

### 5.3 Balancing Accuracy vs. Performance

For a TTS system, it may not be necessary to implement every aspect of the ALBERT architecture at full fidelity. Potential simplifications include:

- Reducing the number of attention heads 
- Limiting the maximum sequence length
- Using fp16 or quantized weights
- Implementing a subset of the layers

These decisions should be guided by quality vs. performance testing.

## 6. Testing Strategy

### 6.1 Component Tests

Each component (attention, feed-forward, etc.) should have unit tests that verify:

- Shape handling for various input dimensions
- Correct mathematical operations
- Weight loading accuracy

### 6.2 Integration Tests

The complete CustomBERT should be tested for:

- End-to-end processing of token sequences
- Comparison with expected output shapes
- Weight loading from binary files

### 6.3 Regression Tests

Once integrated with the full TTS pipeline, test for:

- Performance impact (inference time)
- Memory usage
- Audio quality improvement

## 7. Resources and References

### 7.1 ALBERT Paper and Implementation

- [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)
- [Hugging Face ALBERT Implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/albert)

### 7.2 Relevant Ferrocarril Files

- `ferrocarril/src/model/ferro_model.rs` - Integration point
- `ferrocarril-core/src/lib.rs` - Config structure for ALBERT parameters
- `ferrocarril_weights/model/metadata.json` - Contains weight file locations

### 7.3 Kokoro Reference Files

- `kokoro/kokoro/modules.py` - Contains CustomAlbert implementation
- `kokoro/kokoro/model.py` - Shows how CustomAlbert is used in the pipeline

## 8. Implementation Milestones

1. **Basic Transformer Components** - Implement the fundamental building blocks
2. **CustomBERT Forward Pass** - Get the full CustomBERT working with dummy weights
3. **Weight Loading** - Implement binary weight loading for CustomBERT
4. **Integration** - Connect CustomBERT to the TTS pipeline
5. **Optimization** - Improve performance of the implementation
6. **Testing** - Comprehensive testing and validation

## 9. Conclusion

The CustomBERT component is a critical piece of the Ferrocarril TTS system that will significantly improve output quality. While it's one of the more complex components, its implementation will follow the same pattern as other components in the system. The approach outlined in this document provides a clear path forward, breaking down the task into manageable steps.

The main complexity lies in correctly implementing the transformer architecture in Rust without external dependencies, and ensuring that weight loading and tensor shapes are handled correctly throughout the implementation.