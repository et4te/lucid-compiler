//! Trainer for ML Models
//!
//! This module provides the training loop infrastructure for training
//! neural networks defined in Lucid.

use crate::autodiff::{GradientTape, TapeNodeId, ops};
use crate::data::{DataLoader, Dataset};
use crate::loss::{LossType, compute_loss};
use crate::optimizer_ml::{MLOptimizer, OptimizerConfig, LRScheduler, ParamId};
use crate::tensor::TensorValue;
use ndarray::ArrayD;
use std::collections::HashMap;

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Loss function type
    pub loss_type: LossType,
    /// Optimizer configuration
    pub optimizer: OptimizerConfig,
    /// Learning rate scheduler
    pub lr_scheduler: LRScheduler,
    /// Gradient clipping value (0 = disabled)
    pub grad_clip: f64,
    /// Logging frequency (batches)
    pub log_every: usize,
    /// Validation frequency (epochs)
    pub validate_every: usize,
    /// Early stopping patience (0 = disabled)
    pub early_stopping: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            epochs: 10,
            batch_size: 32,
            loss_type: LossType::CrossEntropy,
            optimizer: OptimizerConfig::adam(0.001),
            lr_scheduler: LRScheduler::Constant,
            grad_clip: 0.0,
            log_every: 100,
            validate_every: 1,
            early_stopping: 0,
        }
    }
}

/// Training metrics for tracking progress
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Training losses per epoch
    pub train_losses: Vec<f64>,
    /// Validation losses per epoch
    pub val_losses: Vec<f64>,
    /// Training accuracy per epoch (if applicable)
    pub train_accuracy: Vec<f64>,
    /// Validation accuracy per epoch (if applicable)
    pub val_accuracy: Vec<f64>,
    /// Learning rate per epoch
    pub learning_rates: Vec<f64>,
}

impl TrainingMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the best validation loss
    pub fn best_val_loss(&self) -> Option<f64> {
        self.val_losses.iter().cloned().reduce(f64::min)
    }

    /// Get the epoch with best validation loss
    pub fn best_epoch(&self) -> Option<usize> {
        self.val_losses
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
    }
}

/// A simple feed-forward model for training
#[derive(Debug)]
pub struct SimpleModel {
    /// Layer weights
    pub weights: Vec<TensorValue>,
    /// Layer biases
    pub biases: Vec<TensorValue>,
    /// Activation function (relu, sigmoid, tanh)
    pub activation: String,
    /// Dropout rate (0 = no dropout)
    pub dropout: f64,
}

impl SimpleModel {
    /// Create a new model with given layer sizes
    pub fn new(layer_sizes: &[usize], activation: &str) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let w = TensorValue::xavier(&[layer_sizes[i], layer_sizes[i + 1]]);
            let b = TensorValue::zeros(&[layer_sizes[i + 1]]);
            weights.push(w);
            biases.push(b);
        }

        SimpleModel {
            weights,
            biases,
            activation: activation.to_string(),
            dropout: 0.0,
        }
    }

    /// Set dropout rate
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Get all parameters as a map
    pub fn parameters(&self) -> HashMap<ParamId, TensorValue> {
        let mut params = HashMap::new();
        for (i, w) in self.weights.iter().enumerate() {
            params.insert(i * 2, w.clone());
        }
        for (i, b) in self.biases.iter().enumerate() {
            params.insert(i * 2 + 1, b.clone());
        }
        params
    }

    /// Update parameters from map
    pub fn update_parameters(&mut self, params: &HashMap<ParamId, TensorValue>) {
        for (i, w) in self.weights.iter_mut().enumerate() {
            if let Some(p) = params.get(&(i * 2)) {
                *w = p.clone();
            }
        }
        for (i, b) in self.biases.iter_mut().enumerate() {
            if let Some(p) = params.get(&(i * 2 + 1)) {
                *b = p.clone();
            }
        }
    }

    /// Forward pass through the model
    pub fn forward(&self, input: &TensorValue, training: bool) -> TensorValue {
        let mut x = input.clone();

        for i in 0..self.weights.len() {
            // Linear: x = x @ W + b
            x = x.matmul(&self.weights[i]);

            // Add bias (broadcast over batch)
            let bias_expanded = TensorValue::new(
                self.biases[i].data.clone().insert_axis(ndarray::Axis(0))
            );
            x = x.add(&bias_expanded);

            // Apply activation (except for last layer)
            if i < self.weights.len() - 1 {
                x = match self.activation.as_str() {
                    "relu" => x.relu(),
                    "sigmoid" => x.sigmoid(),
                    "tanh" => x.tanh(),
                    _ => x,
                };

                // Apply dropout
                if training && self.dropout > 0.0 {
                    x = x.dropout(self.dropout, true);
                }
            }
        }

        x
    }

    /// Forward pass with gradient tape recording
    pub fn forward_with_tape(
        &self,
        tape: &mut GradientTape,
        input_id: TapeNodeId,
    ) -> TapeNodeId {
        let mut x_id = input_id;

        for i in 0..self.weights.len() {
            // Record weight and bias as parameters
            let w_id = tape.record_parameter(self.weights[i].clone());
            let b_id = tape.record_parameter(self.biases[i].clone());

            // Linear: x = x @ W
            let (mm_id, _) = ops::matmul(tape, x_id, w_id);

            // Add bias (simplified - assumes batch dimension)
            let bias_val = tape.get_value(b_id).unwrap().clone();

            // Broadcast bias
            let bias_expanded = TensorValue::new(
                bias_val.data.insert_axis(ndarray::Axis(0))
            );
            let bias_exp_id = tape.record_input(bias_expanded);
            let (add_id, _) = ops::add(tape, mm_id, bias_exp_id);

            x_id = add_id;

            // Activation (except for last layer)
            if i < self.weights.len() - 1 {
                match self.activation.as_str() {
                    "relu" => {
                        let (relu_id, _) = ops::relu(tape, x_id);
                        x_id = relu_id;
                    }
                    _ => {}
                }
            }
        }

        x_id
    }
}

/// Main trainer for training models
#[derive(Debug)]
pub struct Trainer {
    /// Training configuration
    config: TrainingConfig,
    /// Optimizer
    optimizer: MLOptimizer,
    /// Training metrics
    metrics: TrainingMetrics,
    /// Current epoch
    current_epoch: usize,
    /// Global step count
    global_step: usize,
}

impl Trainer {
    /// Create a new trainer with given configuration
    pub fn new(config: TrainingConfig) -> Self {
        let optimizer = MLOptimizer::new(config.optimizer.clone());
        Trainer {
            config,
            optimizer,
            metrics: TrainingMetrics::new(),
            current_epoch: 0,
            global_step: 0,
        }
    }

    /// Train a simple model on a dataset
    pub fn train(
        &mut self,
        model: &mut SimpleModel,
        train_data: &Dataset,
        val_data: Option<&Dataset>,
    ) -> &TrainingMetrics {
        let mut params = model.parameters();
        let mut best_val_loss = f64::INFINITY;
        let mut patience_counter = 0;

        println!("Starting training for {} epochs", self.config.epochs);
        println!("Batch size: {}, Learning rate: {:.6}",
            self.config.batch_size, self.optimizer.learning_rate());

        for epoch in 0..self.config.epochs {
            self.current_epoch = epoch;

            // Update learning rate
            let lr = self.config.lr_scheduler.get_lr(
                self.optimizer.learning_rate(),
                epoch,
                self.global_step,
            );
            self.optimizer.set_learning_rate(lr);
            self.metrics.learning_rates.push(lr);

            // Training epoch
            let train_loss = self.train_epoch(model, &mut params, train_data);
            self.metrics.train_losses.push(train_loss);

            println!("Epoch {}/{}: train_loss = {:.4}, lr = {:.6}",
                epoch + 1, self.config.epochs, train_loss, lr);

            // Validation
            if epoch % self.config.validate_every == 0 {
                if let Some(val) = val_data {
                    let val_loss = self.validate(model, val);
                    self.metrics.val_losses.push(val_loss);
                    println!("  val_loss = {:.4}", val_loss);

                    // Early stopping check
                    if self.config.early_stopping > 0 {
                        if val_loss < best_val_loss {
                            best_val_loss = val_loss;
                            patience_counter = 0;
                        } else {
                            patience_counter += 1;
                            if patience_counter >= self.config.early_stopping {
                                println!("Early stopping at epoch {}", epoch + 1);
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Update model with final parameters
        model.update_parameters(&params);

        &self.metrics
    }

    /// Train for one epoch
    fn train_epoch(
        &mut self,
        model: &SimpleModel,
        params: &mut HashMap<ParamId, TensorValue>,
        data: &Dataset,
    ) -> f64 {
        let mut loader = DataLoader::new(data.clone(), self.config.batch_size, true);
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        while let Some((inputs, targets)) = loader.next_batch() {
            // Forward pass with gradient tape
            let mut tape = GradientTape::new();

            // Record input
            let input_id = tape.record_input(inputs.clone());

            // Forward through model
            let mut x_id = input_id;
            // Store mapping from param key to tape node ID
            let mut param_tape_ids: Vec<(ParamId, usize)> = Vec::new();

            for i in 0..model.weights.len() {
                // Get current parameters
                let w = params.get(&(i * 2)).unwrap();
                let b = params.get(&(i * 2 + 1)).unwrap();

                let w_id = tape.record_parameter(w.clone());
                let _b_id = tape.record_parameter(b.clone());

                // Store both the param key and the tape node ID
                param_tape_ids.push((i * 2, w_id));
                // Note: bias gradient comes from the bias_exp_id, not b_id

                // Linear layer
                let (mm_id, _) = ops::matmul(&mut tape, x_id, w_id);

                // Add bias - broadcast bias to batch dimension
                let bias_exp = TensorValue::new(b.data.clone().insert_axis(ndarray::Axis(0)));
                let bias_exp_id = tape.record_input(bias_exp);
                let (add_id, _) = ops::add(&mut tape, mm_id, bias_exp_id);

                // Store bias gradient mapping - the gradient flows through bias_exp_id
                param_tape_ids.push((i * 2 + 1, bias_exp_id));

                x_id = add_id;

                // Activation
                if i < model.weights.len() - 1 {
                    let (relu_id, _) = ops::relu(&mut tape, x_id);
                    x_id = relu_id;
                }
            }

            // Compute loss
            let target_id = tape.record_input(targets);
            let (loss_id, loss) = ops::cross_entropy_loss(&mut tape, x_id, target_id);

            // Backward pass
            let grads = tape.backward(loss_id);

            // Collect gradients for parameters
            let mut param_grads: HashMap<ParamId, ArrayD<f64>> = HashMap::new();
            for (param_key, tape_id) in &param_tape_ids {
                if let Some(grad) = grads.get(tape_id) {
                    // For bias, sum over batch dimension to get correct shape
                    let grad_for_param = if *param_key % 2 == 1 {
                        // This is a bias - sum over batch dimension
                        grad.sum_axis(ndarray::Axis(0))
                    } else {
                        grad.clone()
                    };
                    param_grads.insert(*param_key, grad_for_param);
                }
            }

            // Gradient clipping
            if self.config.grad_clip > 0.0 {
                for grad in param_grads.values_mut() {
                    let norm: f64 = grad.iter().map(|x| x * x).sum::<f64>().sqrt();
                    if norm > self.config.grad_clip {
                        *grad = grad.mapv(|x| x * self.config.grad_clip / norm);
                    }
                }
            }

            // Update parameters
            self.optimizer.step(params, &param_grads);

            total_loss += loss.to_scalar();
            num_batches += 1;
            self.global_step += 1;

            if self.config.log_every > 0 && self.global_step % self.config.log_every == 0 {
                println!("  Step {}: batch_loss = {:.4}", self.global_step, loss.to_scalar());
            }
        }

        total_loss / num_batches as f64
    }

    /// Validate on a dataset
    fn validate(&self, model: &SimpleModel, data: &Dataset) -> f64 {
        let mut loader = DataLoader::new(data.clone(), self.config.batch_size, false);
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        while let Some((inputs, targets)) = loader.next_batch() {
            // Forward pass (no gradient)
            let outputs = model.forward(&inputs, false);

            // Compute loss
            let (loss, _) = compute_loss(self.config.loss_type, &outputs, &targets);
            total_loss += loss.to_scalar();
            num_batches += 1;
        }

        total_loss / num_batches as f64
    }

    /// Get training metrics
    pub fn metrics(&self) -> &TrainingMetrics {
        &self.metrics
    }

    /// Get current learning rate
    pub fn learning_rate(&self) -> f64 {
        self.optimizer.learning_rate()
    }
}

/// Utility function to compute classification accuracy
pub fn compute_accuracy(predictions: &TensorValue, targets: &TensorValue) -> f64 {
    let pred_classes: Vec<usize> = (0..predictions.shape()[0])
        .map(|i| {
            let mut max_idx = 0;
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..predictions.shape()[1] {
                let val = predictions.data[[i, j]];
                if val > max_val {
                    max_val = val;
                    max_idx = j;
                }
            }
            max_idx
        })
        .collect();

    let target_classes: Vec<usize> = (0..targets.shape()[0])
        .map(|i| {
            let mut max_idx = 0;
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..targets.shape()[1] {
                let val = targets.data[[i, j]];
                if val > max_val {
                    max_val = val;
                    max_idx = j;
                }
            }
            max_idx
        })
        .collect();

    let correct: usize = pred_classes
        .iter()
        .zip(target_classes.iter())
        .filter(|(p, t)| p == t)
        .count();

    correct as f64 / pred_classes.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_model_creation() {
        let model = SimpleModel::new(&[10, 32, 16, 5], "relu");
        assert_eq!(model.weights.len(), 3);
        assert_eq!(model.biases.len(), 3);
    }

    #[test]
    fn test_simple_model_forward() {
        let model = SimpleModel::new(&[10, 5], "relu");
        let input = TensorValue::rand_normal(&[4, 10], 0.0, 1.0);

        let output = model.forward(&input, false);
        assert_eq!(output.shape(), vec![4, 5]);
    }

    #[test]
    fn test_trainer_single_step() {
        let config = TrainingConfig {
            epochs: 1,
            batch_size: 4,
            loss_type: LossType::CrossEntropy,
            optimizer: OptimizerConfig::sgd(0.01),
            log_every: 0,
            ..Default::default()
        };

        let mut trainer = Trainer::new(config);
        let mut model = SimpleModel::new(&[10, 5], "relu");
        let dataset = Dataset::synthetic_classification(16, 10, 5);

        let metrics = trainer.train(&mut model, &dataset, None);
        assert_eq!(metrics.train_losses.len(), 1);
    }

    #[test]
    fn test_accuracy_computation() {
        let predictions = TensorValue::matrix(vec![
            vec![0.9, 0.1, 0.0],  // Predicts class 0
            vec![0.1, 0.8, 0.1],  // Predicts class 1
            vec![0.2, 0.2, 0.6],  // Predicts class 2
        ]);
        let targets = TensorValue::matrix(vec![
            vec![1.0, 0.0, 0.0],  // Class 0
            vec![0.0, 1.0, 0.0],  // Class 1
            vec![0.0, 0.0, 1.0],  // Class 2
        ]);

        let accuracy = compute_accuracy(&predictions, &targets);
        assert!((accuracy - 1.0).abs() < 1e-6); // All correct
    }
}
