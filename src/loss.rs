//! Loss Functions for ML Training
//!
//! This module provides common loss functions used in neural network training.

use crate::tensor::TensorValue;

/// Types of loss functions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LossType {
    /// Mean Squared Error - for regression
    MSE,
    /// Cross Entropy - for classification (expects logits, applies softmax internally)
    CrossEntropy,
    /// Binary Cross Entropy - for binary classification (expects probabilities)
    BinaryCrossEntropy,
    /// Huber Loss - robust to outliers
    Huber { delta: f64 },
}

/// Compute loss and return both the loss value and the gradient w.r.t. predictions
pub fn compute_loss(
    loss_type: LossType,
    predictions: &TensorValue,
    targets: &TensorValue,
) -> (TensorValue, TensorValue) {
    match loss_type {
        LossType::MSE => mse_loss(predictions, targets),
        LossType::CrossEntropy => cross_entropy_loss(predictions, targets),
        LossType::BinaryCrossEntropy => binary_cross_entropy_loss(predictions, targets),
        LossType::Huber { delta } => huber_loss(predictions, targets, delta),
    }
}

/// Mean Squared Error Loss
///
/// MSE = mean((pred - target)^2)
/// Gradient: 2 * (pred - target) / n
pub fn mse_loss(predictions: &TensorValue, targets: &TensorValue) -> (TensorValue, TensorValue) {
    let diff = predictions.sub(targets);
    let squared = diff.mul(&diff);
    let loss = squared.mean();

    // Gradient
    let n = predictions.len() as f64;
    let grad = diff.mul_scalar(2.0 / n);

    (loss, grad)
}

/// Cross Entropy Loss with Softmax
///
/// CE = -mean(sum(target * log(softmax(logits))))
/// Gradient: (softmax(logits) - target) / batch_size
pub fn cross_entropy_loss(logits: &TensorValue, targets: &TensorValue) -> (TensorValue, TensorValue) {
    // Apply softmax for numerical stability
    let probs = logits.softmax();

    // Compute cross-entropy: -sum(target * log(prob + eps))
    let eps = 1e-10;
    let log_probs = probs.add_scalar(eps).log();

    // Element-wise multiply and sum along last axis
    let loss_per_sample = targets.mul(&log_probs);
    let neg_loss = loss_per_sample.sum_axis(loss_per_sample.ndim() - 1);
    let loss = TensorValue::scalar(-neg_loss.mean().to_scalar());

    // Gradient: softmax - targets (averaged over batch)
    let batch_size = logits.shape()[0] as f64;
    let grad = probs.sub(targets).mul_scalar(1.0 / batch_size);

    (loss, grad)
}

/// Binary Cross Entropy Loss
///
/// BCE = -mean(target * log(pred) + (1-target) * log(1-pred))
pub fn binary_cross_entropy_loss(predictions: &TensorValue, targets: &TensorValue) -> (TensorValue, TensorValue) {
    let eps = 1e-10;

    // Clamp predictions to avoid log(0)
    let pred_clamped = TensorValue::new(
        predictions.data.mapv(|x| x.max(eps).min(1.0 - eps))
    );

    // -target * log(pred)
    let term1 = targets.mul(&pred_clamped.log());

    // -(1-target) * log(1-pred)
    let one_minus_target = TensorValue::ones(&targets.shape()).sub(targets);
    let one_minus_pred = TensorValue::ones(&pred_clamped.shape()).sub(&pred_clamped);
    let term2 = one_minus_target.mul(&one_minus_pred.log());

    let neg_loss = term1.add(&term2);
    let loss = TensorValue::scalar(-neg_loss.mean().to_scalar());

    // Gradient: (pred - target) / (pred * (1 - pred))
    let diff = pred_clamped.sub(targets);
    let denom = pred_clamped.mul(&one_minus_pred);
    let n = predictions.len() as f64;
    let grad = diff.div(&denom).mul_scalar(1.0 / n);

    (loss, grad)
}

/// Huber Loss (Smooth L1)
///
/// For |diff| <= delta: 0.5 * diff^2
/// For |diff| > delta: delta * (|diff| - 0.5 * delta)
pub fn huber_loss(predictions: &TensorValue, targets: &TensorValue, delta: f64) -> (TensorValue, TensorValue) {
    let diff = predictions.sub(targets);

    let loss_data = diff.data.mapv(|d| {
        let abs_d = d.abs();
        if abs_d <= delta {
            0.5 * d * d
        } else {
            delta * (abs_d - 0.5 * delta)
        }
    });
    let loss = TensorValue::scalar(loss_data.mean().unwrap_or(0.0));

    // Gradient
    let n = predictions.len() as f64;
    let grad_data = diff.data.mapv(|d| {
        if d.abs() <= delta {
            d / n
        } else {
            delta * d.signum() / n
        }
    });
    let grad = TensorValue::new(grad_data);

    (loss, grad)
}

/// Label smoothing for cross-entropy
///
/// Smoothed targets = (1 - smoothing) * targets + smoothing / num_classes
pub fn label_smoothing(targets: &TensorValue, smoothing: f64) -> TensorValue {
    let num_classes = targets.shape()[targets.ndim() - 1] as f64;
    let smooth_val = smoothing / num_classes;
    let confidence = 1.0 - smoothing;

    targets.mul_scalar(confidence).add_scalar(smooth_val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let pred = TensorValue::vector(vec![1.0, 2.0, 3.0]);
        let target = TensorValue::vector(vec![1.0, 2.0, 3.0]);

        let (loss, _grad) = mse_loss(&pred, &target);
        assert!(loss.to_scalar().abs() < 1e-6); // Perfect prediction = 0 loss
    }

    #[test]
    fn test_mse_loss_nonzero() {
        let pred = TensorValue::vector(vec![0.0, 0.0, 0.0]);
        let target = TensorValue::vector(vec![1.0, 1.0, 1.0]);

        let (loss, _grad) = mse_loss(&pred, &target);
        assert!((loss.to_scalar() - 1.0).abs() < 1e-6); // MSE = 1
    }

    #[test]
    fn test_cross_entropy_loss() {
        // Batch of 2 samples, 3 classes
        let logits = TensorValue::matrix(vec![
            vec![2.0, 1.0, 0.1],  // Should predict class 0
            vec![0.1, 2.0, 0.1],  // Should predict class 1
        ]);
        let targets = TensorValue::matrix(vec![
            vec![1.0, 0.0, 0.0],  // Class 0
            vec![0.0, 1.0, 0.0],  // Class 1
        ]);

        let (loss, grad) = cross_entropy_loss(&logits, &targets);

        // Loss should be small since predictions are correct
        assert!(loss.to_scalar() < 1.0);

        // Gradient should have correct shape
        assert_eq!(grad.shape(), vec![2, 3]);
    }

    #[test]
    fn test_huber_loss() {
        let pred = TensorValue::vector(vec![0.0, 0.0]);
        let target = TensorValue::vector(vec![0.5, 10.0]); // One small, one large diff

        let (loss, _grad) = huber_loss(&pred, &target, 1.0);

        // Should be less than MSE for the outlier
        let (mse_loss, _) = mse_loss(&pred, &target);
        assert!(loss.to_scalar() < mse_loss.to_scalar());
    }

    #[test]
    fn test_label_smoothing() {
        let targets = TensorValue::matrix(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ]);

        let smoothed = label_smoothing(&targets, 0.1);

        // Check that values are smoothed
        // Hot value: 0.9 + 0.1/3 ≈ 0.933
        // Cold value: 0.1/3 ≈ 0.033
        assert!(smoothed.data[[0, 0]] > 0.9);
        assert!(smoothed.data[[0, 1]] > 0.0 && smoothed.data[[0, 1]] < 0.1);
    }
}
