//! Causal Language Model Components
//!
//! This module provides causal (autoregressive) attention and text generation
//! capabilities for building conversational models.

use crate::tensor::TensorValue;
use crate::tokenizer::{Tokenizer, EOS_ID, BOS_ID};
use ndarray::{Array, Array2, ArrayD, Axis, IxDyn, s};
use rand::Rng;
use std::collections::HashMap;

/// Create a causal attention mask
/// Returns a mask where position i can only attend to positions <= i
/// The mask is 0 for allowed positions and -inf for masked positions
pub fn create_causal_mask(seq_len: usize) -> TensorValue {
    let mut mask_data = vec![0.0f64; seq_len * seq_len];

    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                // Future position - mask it out
                mask_data[i * seq_len + j] = f64::NEG_INFINITY;
            }
        }
    }

    TensorValue::new(Array::from_shape_vec(IxDyn(&[seq_len, seq_len]), mask_data).unwrap())
}

/// Create a causal mask for batched inputs
/// Shape: [batch_size, seq_len, seq_len]
pub fn create_batched_causal_mask(batch_size: usize, seq_len: usize) -> TensorValue {
    let single_mask = create_causal_mask(seq_len);
    let mut batched_data = Vec::with_capacity(batch_size * seq_len * seq_len);

    for _ in 0..batch_size {
        batched_data.extend(single_mask.data.iter().cloned());
    }

    TensorValue::new(Array::from_shape_vec(
        IxDyn(&[batch_size, seq_len, seq_len]),
        batched_data
    ).unwrap())
}

/// Causal self-attention layer
#[derive(Debug, Clone)]
pub struct CausalAttention {
    pub query_weights: TensorValue,
    pub key_weights: TensorValue,
    pub value_weights: TensorValue,
    pub output_weights: TensorValue,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl CausalAttention {
    pub fn new(hidden_dim: usize, num_heads: usize) -> Self {
        assert!(hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads");
        let head_dim = hidden_dim / num_heads;

        CausalAttention {
            query_weights: TensorValue::xavier(&[hidden_dim, hidden_dim]),
            key_weights: TensorValue::xavier(&[hidden_dim, hidden_dim]),
            value_weights: TensorValue::xavier(&[hidden_dim, hidden_dim]),
            output_weights: TensorValue::xavier(&[hidden_dim, hidden_dim]),
            hidden_dim,
            num_heads,
            head_dim,
        }
    }

    /// Forward pass with causal masking
    /// Input shape: [batch_size, seq_len, hidden_dim]
    /// Returns: [batch_size, seq_len, hidden_dim]
    pub fn forward(&self, x: &TensorValue) -> TensorValue {
        let shape = x.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let hidden_dim = shape[2];

        // Reshape x from [batch, seq, hidden] to [batch*seq, hidden] for matmul
        let x_2d = x.reshape(&[batch_size * seq_len, hidden_dim]);

        // Project to Q, K, V (2D matmul)
        let q_2d = x_2d.matmul(&self.query_weights);
        let k_2d = x_2d.matmul(&self.key_weights);
        let v_2d = x_2d.matmul(&self.value_weights);

        // Reshape back to [batch, seq, hidden]
        let q = q_2d.reshape(&[batch_size, seq_len, hidden_dim]);
        let k = k_2d.reshape(&[batch_size, seq_len, hidden_dim]);
        let v = v_2d.reshape(&[batch_size, seq_len, hidden_dim]);

        // Compute attention scores: Q @ K^T / sqrt(d_k) using bmm
        let k_t = k.transpose(); // [batch, hidden, seq]
        let scale = (self.head_dim as f64).sqrt();
        let scores = q.bmm(&k_t).scale(1.0 / scale); // [batch, seq, seq]

        // Apply causal mask
        let mask = create_batched_causal_mask(batch_size, seq_len);
        let masked_scores = scores.add(&mask);

        // Softmax along last dimension
        let attn_weights = masked_scores.softmax();

        // Apply attention to values: attn_weights @ v
        let attn_output = attn_weights.bmm(&v); // [batch, seq, hidden]

        // Output projection: reshape to 2D, matmul, reshape back
        let attn_2d = attn_output.reshape(&[batch_size * seq_len, hidden_dim]);
        let output_2d = attn_2d.matmul(&self.output_weights);
        output_2d.reshape(&[batch_size, seq_len, hidden_dim])
    }

    /// Forward pass with KV cache for efficient generation
    /// Returns (output, updated_k_cache, updated_v_cache)
    pub fn forward_with_cache(
        &self,
        x: &TensorValue,
        k_cache: Option<&TensorValue>,
        v_cache: Option<&TensorValue>,
    ) -> (TensorValue, TensorValue, TensorValue) {
        let shape = x.shape();
        let batch_size = shape[0];

        // Project current token
        let q = x.matmul(&self.query_weights);
        let k_new = x.matmul(&self.key_weights);
        let v_new = x.matmul(&self.value_weights);

        // Concatenate with cache if present
        let (k, v) = if let (Some(k_c), Some(v_c)) = (k_cache, v_cache) {
            (k_c.concat(&k_new, 1), v_c.concat(&v_new, 1))
        } else {
            (k_new.clone(), v_new.clone())
        };

        let seq_len = k.shape()[1];

        // Compute attention scores
        let k_t = k.transpose();
        let scale = (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k_t).scale(1.0 / scale);

        // For generation, we only need to mask future positions
        // Since we're generating one token at a time, current position can attend to all previous
        let attn_weights = scores.softmax();

        // Apply attention
        let attn_output = attn_weights.matmul(&v);
        let output = attn_output.matmul(&self.output_weights);

        (output, k, v)
    }

    /// Get parameters for training
    pub fn parameters(&self) -> Vec<&TensorValue> {
        vec![
            &self.query_weights,
            &self.key_weights,
            &self.value_weights,
            &self.output_weights,
        ]
    }
}

/// Feed-forward network with ReLU/GELU activation
#[derive(Debug, Clone)]
pub struct FeedForward {
    pub weights1: TensorValue,
    pub bias1: TensorValue,
    pub weights2: TensorValue,
    pub bias2: TensorValue,
    pub hidden_dim: usize,
    pub ff_dim: usize,
}

impl FeedForward {
    pub fn new(hidden_dim: usize, ff_dim: usize) -> Self {
        FeedForward {
            weights1: TensorValue::xavier(&[hidden_dim, ff_dim]),
            bias1: TensorValue::zeros(&[ff_dim]),
            weights2: TensorValue::xavier(&[ff_dim, hidden_dim]),
            bias2: TensorValue::zeros(&[hidden_dim]),
            hidden_dim,
            ff_dim,
        }
    }

    pub fn forward(&self, x: &TensorValue) -> TensorValue {
        let shape = x.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let hidden_dim = shape[2];

        // Reshape to 2D for matmul: [batch*seq, hidden]
        let x_2d = x.reshape(&[batch_size * seq_len, hidden_dim]);

        // x @ W1 + b1
        let h = x_2d.matmul(&self.weights1).add_bias(&self.bias1);
        // ReLU
        let h = h.relu();
        // h @ W2 + b2
        let out_2d = h.matmul(&self.weights2).add_bias(&self.bias2);

        // Reshape back to 3D
        out_2d.reshape(&[batch_size, seq_len, hidden_dim])
    }

    pub fn parameters(&self) -> Vec<&TensorValue> {
        vec![&self.weights1, &self.bias1, &self.weights2, &self.bias2]
    }
}

/// Transformer decoder block (for causal LM)
#[derive(Debug, Clone)]
pub struct DecoderBlock {
    pub attention: CausalAttention,
    pub ff: FeedForward,
    pub ln1_weight: TensorValue,
    pub ln1_bias: TensorValue,
    pub ln2_weight: TensorValue,
    pub ln2_bias: TensorValue,
    pub hidden_dim: usize,
}

impl DecoderBlock {
    pub fn new(hidden_dim: usize, num_heads: usize, ff_dim: usize) -> Self {
        DecoderBlock {
            attention: CausalAttention::new(hidden_dim, num_heads),
            ff: FeedForward::new(hidden_dim, ff_dim),
            ln1_weight: TensorValue::ones(&[hidden_dim]),
            ln1_bias: TensorValue::zeros(&[hidden_dim]),
            ln2_weight: TensorValue::ones(&[hidden_dim]),
            ln2_bias: TensorValue::zeros(&[hidden_dim]),
            hidden_dim,
        }
    }

    pub fn forward(&self, x: &TensorValue) -> TensorValue {
        // Pre-norm architecture
        // x = x + attention(layer_norm(x))
        let normed = x.layer_norm_with_params(&self.ln1_weight, &self.ln1_bias);
        let attn_out = self.attention.forward(&normed);
        let x = x.add(&attn_out);

        // x = x + ff(layer_norm(x))
        let normed = x.layer_norm_with_params(&self.ln2_weight, &self.ln2_bias);
        let ff_out = self.ff.forward(&normed);
        x.add(&ff_out)
    }
}

/// GPT-style causal language model
#[derive(Debug, Clone)]
pub struct CausalLM {
    /// Token embeddings
    pub token_embeddings: TensorValue,
    /// Position embeddings
    pub position_embeddings: TensorValue,
    /// Decoder blocks
    pub blocks: Vec<DecoderBlock>,
    /// Final layer norm
    pub ln_final_weight: TensorValue,
    pub ln_final_bias: TensorValue,
    /// Output projection (to vocab logits)
    pub output_weights: TensorValue,
    /// Model configuration
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
}

impl CausalLM {
    /// Create a new causal language model
    pub fn new(
        vocab_size: usize,
        hidden_dim: usize,
        num_layers: usize,
        num_heads: usize,
        ff_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            blocks.push(DecoderBlock::new(hidden_dim, num_heads, ff_dim));
        }

        CausalLM {
            token_embeddings: TensorValue::xavier(&[vocab_size, hidden_dim]),
            position_embeddings: TensorValue::xavier(&[max_seq_len, hidden_dim]),
            blocks,
            ln_final_weight: TensorValue::ones(&[hidden_dim]),
            ln_final_bias: TensorValue::zeros(&[hidden_dim]),
            output_weights: TensorValue::xavier(&[hidden_dim, vocab_size]),
            vocab_size,
            hidden_dim,
            num_layers,
            max_seq_len,
        }
    }

    /// Create a tiny model for testing
    pub fn tiny(vocab_size: usize) -> Self {
        Self::new(
            vocab_size,
            64,      // hidden_dim
            2,       // num_layers
            2,       // num_heads
            128,     // ff_dim
            128,     // max_seq_len
        )
    }

    /// Forward pass
    /// Input: [batch_size, seq_len] token IDs
    /// Output: [batch_size, seq_len, vocab_size] logits
    pub fn forward(&self, token_ids: &[Vec<usize>]) -> TensorValue {
        let batch_size = token_ids.len();
        let seq_len = token_ids[0].len();

        // Get embeddings
        let mut x = self.embed(token_ids);

        // Add position embeddings
        let pos_emb = self.get_position_embeddings(seq_len);
        x = x.add(&pos_emb);

        // Pass through decoder blocks
        for block in &self.blocks {
            x = block.forward(&x);
        }

        // Final layer norm
        x = x.layer_norm_with_params(&self.ln_final_weight, &self.ln_final_bias);

        // Project to vocabulary
        // x is [batch, seq, hidden], output_weights is [hidden, vocab]
        // Reshape to 2D, matmul, reshape back
        let shape = x.shape();
        let x_2d = x.reshape(&[shape[0] * shape[1], shape[2]]);
        let logits_2d = x_2d.matmul(&self.output_weights);
        logits_2d.reshape(&[shape[0], shape[1], self.vocab_size])
    }

    fn embed(&self, token_ids: &[Vec<usize>]) -> TensorValue {
        let batch_size = token_ids.len();
        let seq_len = token_ids[0].len();
        let hidden_dim = self.hidden_dim;

        let mut data = vec![0.0; batch_size * seq_len * hidden_dim];

        for (b, seq) in token_ids.iter().enumerate() {
            for (s, &token_id) in seq.iter().enumerate() {
                let token_id = token_id.min(self.vocab_size - 1);
                for h in 0..hidden_dim {
                    let out_idx = b * seq_len * hidden_dim + s * hidden_dim + h;
                    // token_embeddings has shape [vocab_size, hidden_dim]
                    data[out_idx] = self.token_embeddings.data[[token_id, h]];
                }
            }
        }

        TensorValue::new(Array::from_shape_vec(
            IxDyn(&[batch_size, seq_len, hidden_dim]),
            data
        ).unwrap())
    }

    fn get_position_embeddings(&self, seq_len: usize) -> TensorValue {
        let hidden_dim = self.hidden_dim;
        let mut data = vec![0.0; seq_len * hidden_dim];

        for s in 0..seq_len.min(self.max_seq_len) {
            for h in 0..hidden_dim {
                let out_idx = s * hidden_dim + h;
                // position_embeddings has shape [max_seq_len, hidden_dim]
                data[out_idx] = self.position_embeddings.data[[s, h]];
            }
        }

        TensorValue::new(Array::from_shape_vec(
            IxDyn(&[1, seq_len, hidden_dim]),
            data
        ).unwrap())
    }

    /// Compute loss for language modeling (next token prediction)
    pub fn compute_loss(&self, token_ids: &[Vec<usize>]) -> f64 {
        let logits = self.forward(token_ids);
        let batch_size = token_ids.len();
        let seq_len = token_ids[0].len();

        // Cross-entropy loss: -sum(log(softmax(logits)[target]))
        let mut total_loss = 0.0;
        let mut count = 0;

        for b in 0..batch_size {
            for s in 0..seq_len - 1 {
                // Target is next token
                let target = token_ids[b][s + 1];

                // Get logits for this position
                let logit_slice: Vec<f64> = (0..self.vocab_size)
                    .map(|v| {
                        let idx = b * seq_len * self.vocab_size + s * self.vocab_size + v;
                        logits.data.as_slice().unwrap()[idx]
                    })
                    .collect();

                // Softmax
                let max_logit = logit_slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_sum: f64 = logit_slice.iter().map(|&l| (l - max_logit).exp()).sum();
                let log_softmax = logit_slice[target] - max_logit - exp_sum.ln();

                total_loss -= log_softmax;
                count += 1;
            }
        }

        total_loss / count as f64
    }
}

/// Sampling configuration for text generation
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for softmax (higher = more random)
    pub temperature: f64,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Top-p (nucleus) sampling (1.0 = disabled)
    pub top_p: f64,
    /// Repetition penalty (1.0 = disabled)
    pub repetition_penalty: f64,
    /// Maximum tokens to generate
    pub max_tokens: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        SamplingConfig {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            max_tokens: 100,
        }
    }
}

impl SamplingConfig {
    pub fn greedy() -> Self {
        SamplingConfig {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            max_tokens: 100,
        }
    }

    pub fn creative() -> Self {
        SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.95,
            repetition_penalty: 1.2,
            max_tokens: 200,
        }
    }
}

/// Text generator using causal language model
pub struct Generator<T: Tokenizer> {
    pub model: CausalLM,
    pub tokenizer: T,
}

impl<T: Tokenizer> Generator<T> {
    pub fn new(model: CausalLM, tokenizer: T) -> Self {
        Generator { model, tokenizer }
    }

    /// Generate text given a prompt
    pub fn generate(&self, prompt: &str, config: &SamplingConfig) -> String {
        let mut tokens = self.tokenizer.encode(prompt);

        // Add BOS if not present
        if tokens.is_empty() || tokens[0] != BOS_ID {
            tokens.insert(0, BOS_ID);
        }

        let mut rng = rand::thread_rng();

        for _ in 0..config.max_tokens {
            if tokens.len() >= self.model.max_seq_len {
                break;
            }

            // Forward pass
            let logits = self.model.forward(&[tokens.clone()]);

            // Get logits for last position
            let seq_len = tokens.len();
            let vocab_size = self.model.vocab_size;
            let last_logits: Vec<f64> = (0..vocab_size)
                .map(|v| {
                    let idx = (seq_len - 1) * vocab_size + v;
                    logits.data.as_slice().unwrap()[idx]
                })
                .collect();

            // Apply repetition penalty
            let mut logits_vec = last_logits;
            if config.repetition_penalty != 1.0 {
                for &token in &tokens {
                    if token < logits_vec.len() {
                        if logits_vec[token] > 0.0 {
                            logits_vec[token] /= config.repetition_penalty;
                        } else {
                            logits_vec[token] *= config.repetition_penalty;
                        }
                    }
                }
            }

            // Sample next token
            let next_token = self.sample_token(&logits_vec, config, &mut rng);

            if next_token == EOS_ID {
                break;
            }

            tokens.push(next_token);
        }

        self.tokenizer.decode(&tokens)
    }

    fn sample_token(
        &self,
        logits: &[f64],
        config: &SamplingConfig,
        rng: &mut impl Rng,
    ) -> usize {
        let mut probs = logits.to_vec();

        // Apply temperature
        if config.temperature > 0.0 && config.temperature != 1.0 {
            for p in &mut probs {
                *p /= config.temperature;
            }
        }

        // Softmax
        let max_logit = probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut exp_probs: Vec<f64> = probs.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum: f64 = exp_probs.iter().sum();
        for p in &mut exp_probs {
            *p /= sum;
        }

        // Top-k filtering
        if config.top_k > 0 && config.top_k < exp_probs.len() {
            let mut indexed: Vec<(usize, f64)> = exp_probs.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for (i, prob) in &indexed[config.top_k..] {
                exp_probs[*i] = 0.0;
            }
        }

        // Top-p (nucleus) filtering
        if config.top_p < 1.0 {
            let mut indexed: Vec<(usize, f64)> = exp_probs.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let mut cumsum = 0.0;
            for (i, prob) in &indexed {
                cumsum += prob;
                if cumsum > config.top_p {
                    exp_probs[*i] = 0.0;
                }
            }
        }

        // Renormalize
        let sum: f64 = exp_probs.iter().sum();
        if sum > 0.0 {
            for p in &mut exp_probs {
                *p /= sum;
            }
        } else {
            // Fallback to uniform if all probabilities are zero
            let n = exp_probs.len();
            for p in &mut exp_probs {
                *p = 1.0 / n as f64;
            }
        }

        // Greedy decoding
        if config.temperature == 0.0 || config.top_k == 1 {
            return exp_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
        }

        // Sample from distribution
        let r: f64 = rng.gen();
        let mut cumsum = 0.0;
        for (i, &p) in exp_probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }

        exp_probs.len() - 1
    }

    /// Interactive chat loop
    pub fn chat(&self, config: &SamplingConfig) {
        use std::io::{stdin, stdout, Write};

        println!("Chat started. Type 'quit' to exit.");
        println!("-----------------------------------");

        let mut history = String::new();

        loop {
            print!("You: ");
            stdout().flush().unwrap();

            let mut input = String::new();
            stdin().read_line(&mut input).unwrap();
            let input = input.trim();

            if input.to_lowercase() == "quit" {
                break;
            }

            // Build prompt with history
            history.push_str("User: ");
            history.push_str(input);
            history.push_str("\nAssistant: ");

            // Generate response
            let response = self.generate(&history, config);

            // Extract just the new response
            let response = if let Some(pos) = response.rfind("Assistant: ") {
                response[pos + 11..].trim()
            } else {
                response.trim()
            };

            // Truncate at newline or end marker
            let response = response.split('\n').next().unwrap_or(response);

            println!("Bot: {}", response);

            // Update history
            history.push_str(response);
            history.push('\n');

            // Limit history length
            if history.len() > 1000 {
                if let Some(pos) = history[history.len() - 800..].find("User: ") {
                    history = history[history.len() - 800 + pos..].to_string();
                }
            }
        }

        println!("Goodbye!");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::CharTokenizer;

    #[test]
    fn test_causal_mask() {
        let mask = create_causal_mask(4);
        let data = mask.data.as_slice().unwrap();

        // Check diagonal and below are 0
        assert_eq!(data[0], 0.0);  // (0,0)
        assert_eq!(data[4], 0.0);  // (1,0)
        assert_eq!(data[5], 0.0);  // (1,1)

        // Check above diagonal is -inf
        assert!(data[1].is_infinite() && data[1] < 0.0);  // (0,1)
        assert!(data[2].is_infinite() && data[2] < 0.0);  // (0,2)
    }

    #[test]
    fn test_causal_lm_forward() {
        let model = CausalLM::tiny(100);
        let tokens = vec![vec![1, 2, 3, 4]];

        let logits = model.forward(&tokens);

        assert_eq!(logits.shape(), vec![1, 4, 100]);
    }

    #[test]
    fn test_generator() {
        let tokenizer = CharTokenizer::new();
        let model = CausalLM::tiny(tokenizer.vocab_size());
        let generator = Generator::new(model, tokenizer);

        let config = SamplingConfig {
            max_tokens: 10,
            ..Default::default()
        };

        let output = generator.generate("Hello", &config);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_sampling_greedy() {
        let config = SamplingConfig::greedy();
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_k, 1);
    }
}
