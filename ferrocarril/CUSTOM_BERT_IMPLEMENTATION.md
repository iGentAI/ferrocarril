# CustomBERT Implementation for Ferrocarril TTS

This document provides an overview of the CustomBERT implementation that has been added to the Ferrocarril TTS system, following the ALBERT architecture with parameter sharing.

## 1. Implementation Overview

The CustomBERT component has been successfully integrated into the Ferrocarril TTS system. This component provides contextual text encodings that significantly improve the naturalness of synthesized speech. The implementation follows the ALBERT architecture, which is a lighter version of BERT with parameter sharing across layers.

### Key Components Implemented:

1. **Layer Normalization** (`layer_norm.rs`): Implements layer normalization with trainable parameters (gamma and beta) for normalizing hidden states.

2. **Embeddings** (`embeddings.rs`): Provides token, position, and token type embeddings, followed by layer normalization.

3. **Multi-Head Attention** (`attention.rs`): Implements scaled dot-product attention with multiple heads, including query, key, and value projections, as well as output projection with residual connection and layer normalization.

4. **Feed-Forward Network** (`feed_forward.rs`): Implements a position-wise feed-forward network with two linear transformations and a GELU activation function, as used in the original BERT paper.

5. **Transformer Architecture** (`transformer.rs`): Combines the above components into a complete ALBERT-style transformer layer with parameter sharing, allowing the same layer to be applied multiple times for efficiency.

6. **CustomBERT Model** (`transformer.rs`): The main model that integrates all components, including embeddings, encoder layers, and embedding-to-hidden mapping.

### Integration with Ferrocarril:

- The CustomBERT implementation has been integrated into the FerroModel class in `ferrocarril/src/model/ferro_model.rs`
- Weight loading has been implemented to load pre-trained weights from the binary format
- The inference pipeline has been updated to use the CustomBERT component for real contextual embeddings

## 2. Architecture Details

The ALBERT architecture has the following key characteristics:

1. **Embedding Factorization**: The token embeddings project to a smaller intermediate space before being projected to the hidden size, reducing parameters.

2. **Parameter Sharing**: The same transformer layer is applied multiple times, significantly reducing parameters.

3. **Layer Structure**:
   - Multi-head attention with residual connection and layer normalization
   - Feed-forward network with GELU activation, followed by residual connection and layer normalization

## 3. Usage Example

Here's a simple example of how to use the CustomBERT component:

```rust
// Create configuration for BERT
let bert_config = BertConfig {
    vocab_size: config.n_token,
    hidden_size: config.plbert.hidden_size,
    num_attention_heads: config.plbert.num_attention_heads,
    num_hidden_layers: config.plbert.num_hidden_layers,
    intermediate_size: config.plbert.intermediate_size,
    max_position_embeddings: 512,
    dropout_prob: config.dropout,
};

// Initialize CustomBERT model
let bert = CustomBert::new(bert_config);

// Load weights
bert.load_weights_binary(&loader, "bert", "module")?;

// Create input tensors
let input_ids = create_input_ids_tensor(); // [batch_size, seq_len]
let attention_mask = create_attention_mask(); // [batch_size, seq_len, seq_len]

// Forward pass
let hidden_states = bert.forward(&input_ids, None, Some(&attention_mask));

// Project to model's hidden dimension
let bert_encoder = Linear::new(bert_hidden_size, hidden_dim, true);
let projected_states = bert_encoder.forward(&hidden_states);
```

## 4. Weight Loading

The component can load weights from the binary format produced by `weight_converter.py`. The weight paths follow the same structure as the PyTorch model:

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
  ...
```

## 5. Testing

A test file `ferrocarril/tests/custom_bert_test.rs` has been added to validate the CustomBERT implementation, checking:

1. Forward pass output shape
2. Non-zero output values
3. Attention mask functionality

## 6. Next Steps

The currently implemented CustomBERT component meets the requirements for the Ferrocarril TTS system. However, some potential future improvements could include:

1. **Performance Optimization**: Optimize matrix multiplication operations using SIMD instructions or a linear algebra library.

2. **Memory Optimization**: Reuse memory for intermediate tensors to reduce allocations.

3. **Batched Processing**: Improve batch processing for more efficient inference when handling multiple inputs.

4. **Additional Features**: Implement additional ALBERT features such as sentence order prediction for fine-tuning.

## 7. Conclusion

The CustomBERT component is now successfully integrated into the Ferrocarril TTS system, replacing the previous placeholder implementation. This component provides true contextual understanding of input text, which is crucial for generating natural-sounding speech with proper prosody and emphasis.

The implementation follows the ALBERT architecture with parameter sharing, allowing for efficient memory usage while maintaining the capabilities of the larger BERT model. All components are implemented in pure Rust, following the project's goal of having a zero-dependency TTS inference system (except for Phonesis).