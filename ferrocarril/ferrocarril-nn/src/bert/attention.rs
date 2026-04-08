//! Multi-head attention implementation for BERT
//!
//! Direct-slice implementation throughout. All linear projections go
//! through `linear_f32`, the reshape-for-heads is a flat-index
//! gather, attention scores are a contiguous inner dot product,
//! softmax runs in place over contiguous rows, and the final
//! context = attention @ value is again a contiguous loop.
//!
//! The old implementation used `tensor[&[...]]` indexed access
//! everywhere in 4-5-nested loops, which unvectorisably paid ~50 ns
//! per element access. For Kokoro's plbert (hidden=768, heads=12,
//! head_size=64) at BERT's shared 12-layer depth this accounted for
//! roughly ~1 second of the ~1.3 second BERT forward time.

#![allow(dead_code)]

use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::{Parameter, LoadWeightsBinary, FerroError};
use ferrocarril_core::ops::matmul::linear_f32;
use ferrocarril_core::weights_binary::BinaryWeightLoader;
use super::layer_norm::LayerNorm;

/// Linear projection layer for attention. Thin wrapper around
/// `linear_f32` so the four attention projections share one kernel.
struct LinearProjection {
    input_dim: usize,
    output_dim: usize,
    weight: Parameter,
    bias: Parameter,
}

impl LinearProjection {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        let weight = Parameter::new(Tensor::from_data(
            vec![0.0; output_dim * input_dim],
            vec![output_dim, input_dim],
        ));
        let bias = Parameter::new(Tensor::from_data(
            vec![0.0; output_dim],
            vec![output_dim],
        ));

        Self {
            input_dim,
            output_dim,
            weight,
            bias,
        }
    }

    /// Forward pass: `y[B, T, out] = x[B, T, in] @ w^T + bias`
    ///
    /// Internally flattens to `(B*T, in)` × `w^T` → `(B*T, out)` so
    /// the fast `linear_f32` kernel handles everything.
    fn forward(&self, x: &Tensor<f32>) -> Tensor<f32> {
        let shape = x.shape();
        assert_eq!(
            shape.len(),
            3,
            "LinearProjection: expected 3D input [B, T, in], got {:?}",
            shape
        );
        let batch_size = shape[0];
        let seq_len = shape[1];
        let input_dim = shape[2];
        assert_eq!(
            input_dim, self.input_dim,
            "LinearProjection: input dim {} does not match configured {}",
            input_dim, self.input_dim
        );

        let m = batch_size * seq_len;
        let k = self.input_dim;
        let n = self.output_dim;

        let mut out = vec![0.0f32; m * n];
        linear_f32(
            x.data(),
            self.weight.data().data(),
            Some(self.bias.data().data()),
            &mut out,
            m,
            k,
            n,
        );

        Tensor::from_data(out, vec![batch_size, seq_len, n])
    }

    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component_path: &str,
        module_path: &str,
        name: &str,
    ) -> Result<(), FerroError> {
        let weight_path = format!("{}.{}.{}.weight", component_path, module_path, name);
        *self.weight.data_mut() = loader.load_tensor(&weight_path)?;

        let bias_path = format!("{}.{}.{}.bias", component_path, module_path, name);
        *self.bias.data_mut() = loader.load_tensor(&bias_path)?;

        Ok(())
    }
}

/// Multi-head attention module for BERT
pub struct MultiHeadAttention {
    hidden_size: usize,
    num_attention_heads: usize,
    attention_head_size: usize,
    query: LinearProjection,
    key: LinearProjection,
    value: LinearProjection,
    output: LinearProjection,
    layer_norm: LayerNorm,
    dropout_prob: f32,
}

impl MultiHeadAttention {
    pub fn new(hidden_size: usize, num_attention_heads: usize, dropout_prob: f32) -> Self {
        assert!(
            hidden_size % num_attention_heads == 0,
            "Hidden size ({}) must be divisible by the number of attention heads ({})",
            hidden_size,
            num_attention_heads
        );

        let attention_head_size = hidden_size / num_attention_heads;

        let query = LinearProjection::new(hidden_size, hidden_size);
        let key = LinearProjection::new(hidden_size, hidden_size);
        let value = LinearProjection::new(hidden_size, hidden_size);
        let output = LinearProjection::new(hidden_size, hidden_size);
        let layer_norm = LayerNorm::new(hidden_size, 1e-12);

        Self {
            hidden_size,
            num_attention_heads,
            attention_head_size,
            query,
            key,
            value,
            output,
            layer_norm,
            dropout_prob,
        }
    }

    /// Forward pass for multi-head attention.
    ///
    /// `hidden_states` has shape `[B, T, hidden]`. The optional
    /// `attention_mask` is either `[B, T]` or `[B, Q, K]` with HF
    /// convention (1 = visible, 0 = masked).
    pub fn forward(
        &self,
        hidden_states: &Tensor<f32>,
        attention_mask: Option<&Tensor<i64>>,
    ) -> Tensor<f32> {
        let shape = hidden_states.shape();
        assert_eq!(shape.len(), 3, "MHA: expected 3D input [B, T, H], got {:?}", shape);
        let batch = shape[0];
        let seq = shape[1];
        let hidden = shape[2];
        assert_eq!(
            hidden, self.hidden_size,
            "MHA: input hidden {} does not match configured {}",
            hidden, self.hidden_size
        );

        let heads = self.num_attention_heads;
        let head_size = self.attention_head_size;

        // 1. Q, K, V projections via linear_f32.
        //    Shapes after: [B, T, hidden]
        let q_proj = self.query.forward(hidden_states);
        let k_proj = self.key.forward(hidden_states);
        let v_proj = self.value.forward(hidden_states);

        let q = q_proj.data();
        let k = k_proj.data();
        let v = v_proj.data();

        // 2. Reshape for heads:
        //    Source:  (B, T, heads*head_size)
        //    Target:  (B, heads, T, head_size)
        //    Flat index: (((b * heads + h) * T) + s) * head_size + d
        //               ← ((b * T + s) * (heads*head_size)) + h*head_size + d
        //
        // We do this as a direct slice reshuffle.
        let per_row = hidden; // = heads * head_size
        let q_bhsd_len = batch * heads * seq * head_size;
        let mut q_bhsd = vec![0.0f32; q_bhsd_len];
        let mut k_bhsd = vec![0.0f32; q_bhsd_len];
        let mut v_bhsd = vec![0.0f32; q_bhsd_len];

        for b in 0..batch {
            for s in 0..seq {
                let src_off = (b * seq + s) * per_row;
                for h in 0..heads {
                    let dst_off = (((b * heads + h) * seq) + s) * head_size;
                    let src_start = src_off + h * head_size;
                    q_bhsd[dst_off..dst_off + head_size]
                        .copy_from_slice(&q[src_start..src_start + head_size]);
                    k_bhsd[dst_off..dst_off + head_size]
                        .copy_from_slice(&k[src_start..src_start + head_size]);
                    v_bhsd[dst_off..dst_off + head_size]
                        .copy_from_slice(&v[src_start..src_start + head_size]);
                }
            }
        }

        // 3. Attention scores:
        //    scores[b, h, q_pos, k_pos] = (Σ_d q[b, h, q_pos, d] * k[b, h, k_pos, d]) / sqrt(head_size)
        //
        //    Shape: [B, heads, seq, seq]
        let scale = 1.0f32 / (head_size as f32).sqrt();
        let mut scores = vec![0.0f32; batch * heads * seq * seq];

        for bh in 0..(batch * heads) {
            let q_off = bh * seq * head_size;
            let k_off = bh * seq * head_size;
            let s_off = bh * seq * seq;

            for q_pos in 0..seq {
                let q_row = &q_bhsd[q_off + q_pos * head_size..q_off + (q_pos + 1) * head_size];
                for k_pos in 0..seq {
                    let k_row =
                        &k_bhsd[k_off + k_pos * head_size..k_off + (k_pos + 1) * head_size];
                    // Contiguous dot product — LLVM vectorises this.
                    let mut acc = 0.0f32;
                    for d in 0..head_size {
                        acc += q_row[d] * k_row[d];
                    }
                    scores[s_off + q_pos * seq + k_pos] = acc * scale;
                }
            }
        }

        // 4. Apply attention mask (HF convention: 1 = visible).
        if let Some(mask) = attention_mask {
            let mask_shape = mask.shape();
            let mask_dims = mask_shape.len();
            let mask_data = mask.data();
            for b in 0..batch {
                for h in 0..heads {
                    let s_off = ((b * heads + h) * seq) * seq;
                    for q_pos in 0..seq {
                        for k_pos in 0..seq {
                            let visible: bool = match mask_dims {
                                2 => mask_data[b * mask_shape[1] + k_pos] > 0,
                                3 => {
                                    let q_bound = mask_shape[1];
                                    let k_bound = mask_shape[2];
                                    if q_pos < q_bound && k_pos < k_bound {
                                        mask_data
                                            [(b * q_bound + q_pos) * k_bound + k_pos]
                                            > 0
                                    } else {
                                        true
                                    }
                                }
                                _ => true,
                            };
                            if !visible {
                                scores[s_off + q_pos * seq + k_pos] = -10000.0;
                            }
                        }
                    }
                }
            }
        }

        // 5. Softmax over the last dim (each q_pos row of seq k_pos scores).
        for bh in 0..(batch * heads) {
            let s_off = bh * seq * seq;
            for q_pos in 0..seq {
                let row = &mut scores[s_off + q_pos * seq..s_off + (q_pos + 1) * seq];

                // row max for numerical stability
                let mut row_max = f32::NEG_INFINITY;
                for &v in row.iter() {
                    if v > row_max {
                        row_max = v;
                    }
                }

                // exp and sum
                let mut row_sum = 0.0f32;
                for v in row.iter_mut() {
                    let e = (*v - row_max).exp();
                    *v = e;
                    row_sum += e;
                }

                // normalise
                let inv_sum = 1.0f32 / row_sum;
                for v in row.iter_mut() {
                    *v *= inv_sum;
                }
            }
        }
        let attention_probs = &scores;

        // 6. Context = attention @ value:
        //    context[b, h, q_pos, d] = Σ_k attention[b, h, q_pos, k_pos] * v[b, h, k_pos, d]
        let mut context_bhsd = vec![0.0f32; batch * heads * seq * head_size];
        for bh in 0..(batch * heads) {
            let s_off = bh * seq * seq;
            let v_off = bh * seq * head_size;
            let c_off = bh * seq * head_size;

            for q_pos in 0..seq {
                let attn_row = &attention_probs[s_off + q_pos * seq..s_off + (q_pos + 1) * seq];
                let c_row = &mut context_bhsd
                    [c_off + q_pos * head_size..c_off + (q_pos + 1) * head_size];

                // Initialise to zero and accumulate weighted values.
                for x in c_row.iter_mut() {
                    *x = 0.0;
                }
                for k_pos in 0..seq {
                    let weight = attn_row[k_pos];
                    let v_row =
                        &v_bhsd[v_off + k_pos * head_size..v_off + (k_pos + 1) * head_size];
                    // Contiguous axpy — vectorised.
                    for d in 0..head_size {
                        c_row[d] += weight * v_row[d];
                    }
                }
            }
        }

        // 7. Reshape context back:
        //    (B, heads, T, head_size) → (B, T, heads*head_size)
        let mut context_flat = vec![0.0f32; batch * seq * hidden];
        for b in 0..batch {
            for s in 0..seq {
                let dst_off = (b * seq + s) * hidden;
                for h in 0..heads {
                    let src_off = (((b * heads + h) * seq) + s) * head_size;
                    let dst_start = dst_off + h * head_size;
                    context_flat[dst_start..dst_start + head_size]
                        .copy_from_slice(&context_bhsd[src_off..src_off + head_size]);
                }
            }
        }
        let context_tensor = Tensor::from_data(context_flat, vec![batch, seq, hidden]);

        // 8. Output projection via the fast linear path.
        let attention_output = self.output.forward(&context_tensor);

        // 9. Residual + LayerNorm (direct slice access).
        let mut residual = attention_output.data().to_vec();
        let input_data = hidden_states.data();
        debug_assert_eq!(residual.len(), input_data.len());
        for (r, &x) in residual.iter_mut().zip(input_data.iter()) {
            *r += x;
        }
        let residual_tensor = Tensor::from_data(residual, vec![batch, seq, hidden]);

        self.layer_norm.forward(&residual_tensor)
    }
}

impl LoadWeightsBinary for MultiHeadAttention {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component_path: &str,
        module_path: &str,
    ) -> Result<(), FerroError> {
        self.query
            .load_weights_binary(loader, component_path, module_path, "query")
            .map_err(|e| FerroError::new(format!("Failed to load query weights: {}", e)))?;

        self.key
            .load_weights_binary(loader, component_path, module_path, "key")
            .map_err(|e| FerroError::new(format!("Failed to load key weights: {}", e)))?;

        self.value
            .load_weights_binary(loader, component_path, module_path, "value")
            .map_err(|e| FerroError::new(format!("Failed to load value weights: {}", e)))?;

        self.output
            .load_weights_binary(loader, component_path, module_path, "dense")
            .map_err(|e| FerroError::new(format!("Failed to load output weights: {}", e)))?;

        self.layer_norm
            .load_weights_binary(loader, component_path, module_path)
            .map_err(|e| FerroError::new(format!("Failed to load layer norm weights: {}", e)))?;

        Ok(())
    }
}