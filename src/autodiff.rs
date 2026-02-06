//! Automatic Differentiation for ML Training
//!
//! This module implements reverse-mode automatic differentiation (backpropagation)
//! using a gradient tape to record operations during forward pass and compute
//! gradients during backward pass.

use crate::tensor::TensorValue;
use ndarray::{ArrayD, IxDyn, Axis, Array};
use std::collections::HashMap;

/// Unique identifier for nodes in the computation graph
pub type TapeNodeId = usize;

/// Type of operation recorded in the tape
#[derive(Debug, Clone)]
pub enum TapeOp {
    // Leaf nodes (no backward needed for inputs)
    Input,
    Parameter,

    // Binary operations
    Add { left: TapeNodeId, right: TapeNodeId },
    Sub { left: TapeNodeId, right: TapeNodeId },
    Mul { left: TapeNodeId, right: TapeNodeId },
    Div { left: TapeNodeId, right: TapeNodeId },
    MatMul { left: TapeNodeId, right: TapeNodeId },

    // Unary operations
    Neg { input: TapeNodeId },
    Relu { input: TapeNodeId },
    Sigmoid { input: TapeNodeId },
    Tanh { input: TapeNodeId },
    Exp { input: TapeNodeId },
    Log { input: TapeNodeId },
    Sqrt { input: TapeNodeId },

    // Reduction operations
    Sum { input: TapeNodeId },
    SumAxis { input: TapeNodeId, axis: usize },
    Mean { input: TapeNodeId },
    MeanAxis { input: TapeNodeId, axis: usize },

    // Shape operations
    Transpose { input: TapeNodeId },
    Reshape { input: TapeNodeId, original_shape: Vec<usize> },

    // Activation functions
    Softmax { input: TapeNodeId },
    LayerNorm { input: TapeNodeId, eps: f64 },

    // Loss functions (combined with softmax for numerical stability)
    CrossEntropyLoss { logits: TapeNodeId, targets: TapeNodeId },
    MSELoss { predicted: TapeNodeId, target: TapeNodeId },
}

/// Entry in the gradient tape
#[derive(Debug, Clone)]
struct TapeEntry {
    /// The operation that produced this value
    op: TapeOp,
    /// The output value (stored for backward pass)
    value: TensorValue,
    /// Shape of the output
    shape: Vec<usize>,
}

/// Gradient tape for recording operations and computing gradients
#[derive(Debug)]
pub struct GradientTape {
    /// Recorded operations
    entries: Vec<TapeEntry>,
    /// Mapping from node IDs to their computed gradients
    gradients: HashMap<TapeNodeId, ArrayD<f64>>,
    /// Whether the tape is currently recording
    recording: bool,
}

impl GradientTape {
    /// Create a new gradient tape
    pub fn new() -> Self {
        GradientTape {
            entries: Vec::new(),
            gradients: HashMap::new(),
            recording: true,
        }
    }

    /// Start recording operations
    pub fn start_recording(&mut self) {
        self.recording = true;
    }

    /// Stop recording operations
    pub fn stop_recording(&mut self) {
        self.recording = false;
    }

    /// Clear the tape
    pub fn clear(&mut self) {
        self.entries.clear();
        self.gradients.clear();
    }

    /// Record an input (leaf node)
    pub fn record_input(&mut self, value: TensorValue) -> TapeNodeId {
        let id = self.entries.len();
        self.entries.push(TapeEntry {
            op: TapeOp::Input,
            shape: value.shape(),
            value,
        });
        id
    }

    /// Record a parameter (leaf node that needs gradient)
    pub fn record_parameter(&mut self, value: TensorValue) -> TapeNodeId {
        let id = self.entries.len();
        self.entries.push(TapeEntry {
            op: TapeOp::Parameter,
            shape: value.shape(),
            value,
        });
        id
    }

    /// Record an operation and return the node ID
    pub fn record_op(&mut self, op: TapeOp, value: TensorValue) -> TapeNodeId {
        if !self.recording {
            return 0; // Return dummy ID when not recording
        }

        let id = self.entries.len();
        self.entries.push(TapeEntry {
            op,
            shape: value.shape(),
            value,
        });
        id
    }

    /// Get the value at a node
    pub fn get_value(&self, id: TapeNodeId) -> Option<&TensorValue> {
        self.entries.get(id).map(|e| &e.value)
    }

    /// Compute gradients via backpropagation
    ///
    /// # Arguments
    /// * `output_id` - The node ID of the loss/output to differentiate
    ///
    /// # Returns
    /// HashMap mapping node IDs to their gradients
    pub fn backward(&mut self, output_id: TapeNodeId) -> &HashMap<TapeNodeId, ArrayD<f64>> {
        self.gradients.clear();

        // Initialize gradient of output with ones (dL/dL = 1)
        let output_shape = self.entries[output_id].shape.clone();
        let initial_grad = if output_shape.is_empty() {
            Array::from_elem(IxDyn(&[]), 1.0)
        } else {
            Array::ones(IxDyn(&output_shape))
        };
        self.gradients.insert(output_id, initial_grad);

        // Backpropagate in reverse order
        for id in (0..=output_id).rev() {
            // Get the gradient flowing into this node
            let grad = match self.gradients.get(&id) {
                Some(g) => g.clone(),
                None => continue, // No gradient flows to this node
            };

            // Clone the operation and necessary data to avoid borrowing issues
            let op = self.entries[id].op.clone();
            let entry_value_data = self.entries[id].value.data.clone();

            // Compute gradients for inputs based on operation type
            match op {
                TapeOp::Input | TapeOp::Parameter => {
                    // Leaf nodes - gradient is already stored
                }

                TapeOp::Add { left, right } => {
                    // d(a+b)/da = 1, d(a+b)/db = 1
                    self.accumulate_grad(left, grad.clone());
                    self.accumulate_grad(right, grad);
                }

                TapeOp::Sub { left, right } => {
                    // d(a-b)/da = 1, d(a-b)/db = -1
                    self.accumulate_grad(left, grad.clone());
                    self.accumulate_grad(right, -grad);
                }

                TapeOp::Mul { left, right } => {
                    // d(a*b)/da = b, d(a*b)/db = a
                    let left_val = self.entries[left].value.data.clone();
                    let right_val = self.entries[right].value.data.clone();
                    self.accumulate_grad(left, &grad * &right_val);
                    self.accumulate_grad(right, &grad * &left_val);
                }

                TapeOp::Div { left, right } => {
                    // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
                    let left_val = self.entries[left].value.data.clone();
                    let right_val = self.entries[right].value.data.clone();
                    self.accumulate_grad(left, &grad / &right_val);
                    self.accumulate_grad(right, -&grad * &left_val / (&right_val * &right_val));
                }

                TapeOp::MatMul { left, right } => {
                    // d(A@B)/dA = grad @ B^T
                    // d(A@B)/dB = A^T @ grad
                    let left_val = self.entries[left].value.clone();
                    let right_val = self.entries[right].value.clone();

                    let grad_tensor = TensorValue::new(grad);
                    let right_t = right_val.transpose();
                    let left_t = left_val.transpose();

                    let grad_left = grad_tensor.matmul(&right_t);
                    let grad_right = left_t.matmul(&grad_tensor);

                    self.accumulate_grad(left, grad_left.data);
                    self.accumulate_grad(right, grad_right.data);
                }

                TapeOp::Neg { input } => {
                    self.accumulate_grad(input, -grad);
                }

                TapeOp::Relu { input } => {
                    // d(relu(x))/dx = 1 if x > 0 else 0
                    let input_val = self.entries[input].value.data.clone();
                    let mask = input_val.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                    self.accumulate_grad(input, &grad * &mask);
                }

                TapeOp::Sigmoid { input } => {
                    // d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
                    let grad_input = &grad * &entry_value_data * &(1.0 - &entry_value_data);
                    self.accumulate_grad(input, grad_input);
                }

                TapeOp::Tanh { input } => {
                    // d(tanh(x))/dx = 1 - tanh(x)^2
                    let grad_input = &grad * &(1.0 - &entry_value_data * &entry_value_data);
                    self.accumulate_grad(input, grad_input);
                }

                TapeOp::Exp { input } => {
                    // d(exp(x))/dx = exp(x)
                    self.accumulate_grad(input, &grad * &entry_value_data);
                }

                TapeOp::Log { input } => {
                    // d(log(x))/dx = 1/x
                    let input_val = self.entries[input].value.data.clone();
                    self.accumulate_grad(input, &grad / &input_val);
                }

                TapeOp::Sqrt { input } => {
                    // d(sqrt(x))/dx = 0.5 / sqrt(x)
                    self.accumulate_grad(input, &grad * 0.5 / &entry_value_data);
                }

                TapeOp::Sum { input } => {
                    // Gradient broadcasts back to input shape
                    let input_shape = self.entries[input].shape.clone();
                    let broadcasted = Array::from_elem(IxDyn(&input_shape), grad[[]]);
                    self.accumulate_grad(input, broadcasted);
                }

                TapeOp::SumAxis { input, axis } => {
                    // Gradient broadcasts along the summed axis
                    let input_shape = self.entries[input].shape.clone();
                    let expanded = grad.insert_axis(Axis(axis));
                    let broadcasted = expanded.broadcast(IxDyn(&input_shape)).unwrap().to_owned();
                    self.accumulate_grad(input, broadcasted);
                }

                TapeOp::Mean { input } => {
                    let input_shape = self.entries[input].shape.clone();
                    let n: usize = input_shape.iter().product();
                    let broadcasted = Array::from_elem(IxDyn(&input_shape), grad[[]] / n as f64);
                    self.accumulate_grad(input, broadcasted);
                }

                TapeOp::MeanAxis { input, axis } => {
                    let input_shape = self.entries[input].shape.clone();
                    let n = input_shape[axis] as f64;
                    let expanded = grad.insert_axis(Axis(axis));
                    let broadcasted = expanded.broadcast(IxDyn(&input_shape)).unwrap().to_owned();
                    self.accumulate_grad(input, broadcasted / n);
                }

                TapeOp::Transpose { input } => {
                    // Transpose the gradient
                    let grad_tensor = TensorValue::new(grad);
                    let grad_t = grad_tensor.transpose();
                    self.accumulate_grad(input, grad_t.data);
                }

                TapeOp::Reshape { input, original_shape } => {
                    // Reshape gradient back to original shape
                    let grad_reshaped = grad.into_shape(IxDyn(&original_shape)).unwrap();
                    self.accumulate_grad(input, grad_reshaped);
                }

                TapeOp::Softmax { input } => {
                    // Softmax backward: grad_input = softmax * (grad - sum(grad * softmax))
                    let last_axis = entry_value_data.ndim() - 1;

                    let grad_times_output = &grad * &entry_value_data;
                    let sum_grad_output = grad_times_output.sum_axis(Axis(last_axis));
                    let sum_expanded = sum_grad_output.insert_axis(Axis(last_axis));

                    let grad_input = &entry_value_data * (&grad - &sum_expanded);
                    self.accumulate_grad(input, grad_input);
                }

                TapeOp::LayerNorm { input, eps } => {
                    // Simplified layer norm backward
                    let input_val = self.entries[input].value.data.clone();
                    let last_axis = input_val.ndim() - 1;
                    let n = input_val.shape()[last_axis] as f64;

                    let mean = input_val.mean_axis(Axis(last_axis)).unwrap().insert_axis(Axis(last_axis));
                    let centered = &input_val - &mean;
                    let var = (&centered * &centered).mean_axis(Axis(last_axis)).unwrap().insert_axis(Axis(last_axis));
                    let std = var.mapv(|x| (x + eps).sqrt());

                    let grad_input = &grad / &std -
                        (&grad * &centered / (&std * &std * &std)).mean_axis(Axis(last_axis)).unwrap().insert_axis(Axis(last_axis)) * &centered / n -
                        grad.mean_axis(Axis(last_axis)).unwrap().insert_axis(Axis(last_axis)) / &std / n;

                    self.accumulate_grad(input, grad_input);
                }

                TapeOp::CrossEntropyLoss { logits, targets } => {
                    // For cross-entropy with softmax: grad = softmax - targets
                    let logits_val = self.entries[logits].value.clone();
                    let targets_val = self.entries[targets].value.data.clone();
                    let softmax = logits_val.softmax();

                    // Scale by the incoming gradient (usually 1/batch_size)
                    let batch_size = logits_val.shape()[0] as f64;
                    let grad_logits = (&softmax.data - &targets_val) * grad[[]] / batch_size;

                    self.accumulate_grad(logits, grad_logits);
                    // No gradient for targets (they're labels)
                }

                TapeOp::MSELoss { predicted, target } => {
                    // d(MSE)/d_pred = 2 * (pred - target) / n
                    let pred_val = self.entries[predicted].value.data.clone();
                    let target_val = self.entries[target].value.data.clone();
                    let n = pred_val.len() as f64;

                    let grad_pred = (&pred_val - &target_val) * 2.0 / n * grad[[]];
                    self.accumulate_grad(predicted, grad_pred);
                    // No gradient for target
                }
            }
        }

        &self.gradients
    }

    /// Accumulate gradient for a node
    fn accumulate_grad(&mut self, node_id: TapeNodeId, grad: ArrayD<f64>) {
        self.gradients
            .entry(node_id)
            .and_modify(|existing| *existing = &*existing + &grad)
            .or_insert(grad);
    }

    /// Get the gradient for a specific node
    pub fn get_gradient(&self, node_id: TapeNodeId) -> Option<&ArrayD<f64>> {
        self.gradients.get(&node_id)
    }

    /// Check if a node is a parameter (needs gradient)
    pub fn is_parameter(&self, node_id: TapeNodeId) -> bool {
        matches!(self.entries.get(node_id).map(|e| &e.op), Some(TapeOp::Parameter))
    }
}

impl Default for GradientTape {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper functions for recording operations with the tape
pub mod ops {
    use super::*;

    pub fn add(tape: &mut GradientTape, left: TapeNodeId, right: TapeNodeId) -> (TapeNodeId, TensorValue) {
        let left_val = tape.get_value(left).unwrap();
        let right_val = tape.get_value(right).unwrap();
        let result = left_val.add(right_val);
        let id = tape.record_op(TapeOp::Add { left, right }, result.clone());
        (id, result)
    }

    pub fn sub(tape: &mut GradientTape, left: TapeNodeId, right: TapeNodeId) -> (TapeNodeId, TensorValue) {
        let left_val = tape.get_value(left).unwrap();
        let right_val = tape.get_value(right).unwrap();
        let result = left_val.sub(right_val);
        let id = tape.record_op(TapeOp::Sub { left, right }, result.clone());
        (id, result)
    }

    pub fn mul(tape: &mut GradientTape, left: TapeNodeId, right: TapeNodeId) -> (TapeNodeId, TensorValue) {
        let left_val = tape.get_value(left).unwrap();
        let right_val = tape.get_value(right).unwrap();
        let result = left_val.mul(right_val);
        let id = tape.record_op(TapeOp::Mul { left, right }, result.clone());
        (id, result)
    }

    pub fn matmul(tape: &mut GradientTape, left: TapeNodeId, right: TapeNodeId) -> (TapeNodeId, TensorValue) {
        let left_val = tape.get_value(left).unwrap();
        let right_val = tape.get_value(right).unwrap();
        let result = left_val.matmul(right_val);
        let id = tape.record_op(TapeOp::MatMul { left, right }, result.clone());
        (id, result)
    }

    pub fn relu(tape: &mut GradientTape, input: TapeNodeId) -> (TapeNodeId, TensorValue) {
        let input_val = tape.get_value(input).unwrap();
        let result = input_val.relu();
        let id = tape.record_op(TapeOp::Relu { input }, result.clone());
        (id, result)
    }

    pub fn softmax(tape: &mut GradientTape, input: TapeNodeId) -> (TapeNodeId, TensorValue) {
        let input_val = tape.get_value(input).unwrap();
        let result = input_val.softmax();
        let id = tape.record_op(TapeOp::Softmax { input }, result.clone());
        (id, result)
    }

    pub fn cross_entropy_loss(tape: &mut GradientTape, logits: TapeNodeId, targets: TapeNodeId) -> (TapeNodeId, TensorValue) {
        let logits_val = tape.get_value(logits).unwrap();
        let targets_val = tape.get_value(targets).unwrap();

        // Compute softmax
        let probs = logits_val.softmax();

        // Compute cross-entropy: -sum(target * log(prob))
        let log_probs = probs.log();
        let loss_per_sample = -(&targets_val.data * &log_probs.data).sum_axis(Axis(1));
        let loss = TensorValue::scalar(loss_per_sample.mean().unwrap_or(0.0));

        let id = tape.record_op(TapeOp::CrossEntropyLoss { logits, targets }, loss.clone());
        (id, loss)
    }

    pub fn mse_loss(tape: &mut GradientTape, predicted: TapeNodeId, target: TapeNodeId) -> (TapeNodeId, TensorValue) {
        let pred_val = tape.get_value(predicted).unwrap();
        let target_val = tape.get_value(target).unwrap();

        let diff = pred_val.sub(target_val);
        let squared = diff.mul(&diff);
        let loss = squared.mean();

        let id = tape.record_op(TapeOp::MSELoss { predicted, target }, loss.clone());
        (id, loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_tape_basic() {
        let mut tape = GradientTape::new();

        // Create inputs
        let a = TensorValue::scalar(3.0);
        let b = TensorValue::scalar(4.0);

        let a_id = tape.record_parameter(a);
        let b_id = tape.record_parameter(b);

        // Compute a * b
        let (prod_id, _) = ops::mul(&mut tape, a_id, b_id);

        // Backward
        tape.backward(prod_id);

        // d(a*b)/da = b = 4
        // d(a*b)/db = a = 3
        let grad_a = tape.get_gradient(a_id).unwrap();
        let grad_b = tape.get_gradient(b_id).unwrap();

        assert!((grad_a[[]] - 4.0).abs() < 1e-6);
        assert!((grad_b[[]] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_chain_rule() {
        let mut tape = GradientTape::new();

        // f(x) = (x + 2) * 3
        // df/dx = 3
        let x = TensorValue::scalar(5.0);
        let two = TensorValue::scalar(2.0);
        let three = TensorValue::scalar(3.0);

        let x_id = tape.record_parameter(x);
        let two_id = tape.record_input(two);
        let three_id = tape.record_input(three);

        let (sum_id, _) = ops::add(&mut tape, x_id, two_id);
        let (prod_id, _) = ops::mul(&mut tape, sum_id, three_id);

        tape.backward(prod_id);

        let grad_x = tape.get_gradient(x_id).unwrap();
        assert!((grad_x[[]] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_relu_gradient() {
        let mut tape = GradientTape::new();

        let x = TensorValue::vector(vec![-1.0, 0.0, 1.0, 2.0]);
        let x_id = tape.record_parameter(x);

        let (relu_id, relu_out) = ops::relu(&mut tape, x_id);

        // Sum to get scalar output
        let sum_id = tape.record_op(TapeOp::Sum { input: relu_id }, relu_out.sum());

        tape.backward(sum_id);

        let grad_x = tape.get_gradient(x_id).unwrap();

        // Gradient should be [0, 0, 1, 1] (relu derivative)
        assert!((grad_x[[0]] - 0.0).abs() < 1e-6);
        assert!((grad_x[[1]] - 0.0).abs() < 1e-6);
        assert!((grad_x[[2]] - 1.0).abs() < 1e-6);
        assert!((grad_x[[3]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_gradient() {
        let mut tape = GradientTape::new();

        // A @ B where A is 2x3, B is 3x2
        let a = TensorValue::matrix(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        let b = TensorValue::matrix(vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ]);

        let a_id = tape.record_parameter(a);
        let b_id = tape.record_parameter(b);

        let (prod_id, prod) = ops::matmul(&mut tape, a_id, b_id);

        // Sum all elements to get scalar loss
        let loss_id = tape.record_op(TapeOp::Sum { input: prod_id }, prod.sum());

        tape.backward(loss_id);

        // Verify gradients exist
        assert!(tape.get_gradient(a_id).is_some());
        assert!(tape.get_gradient(b_id).is_some());
    }
}
