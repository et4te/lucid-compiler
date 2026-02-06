//! Optimizers for ML Training
//!
//! This module provides optimization algorithms for updating neural network parameters.

use crate::tensor::TensorValue;
use ndarray::ArrayD;
use std::collections::HashMap;

/// Parameter identifier
pub type ParamId = usize;

/// Optimizer configuration
#[derive(Debug, Clone)]
pub enum OptimizerConfig {
    /// Stochastic Gradient Descent
    SGD {
        lr: f64,
        momentum: f64,
        weight_decay: f64,
        nesterov: bool,
    },
    /// Adam optimizer
    Adam {
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    },
    /// AdamW (Adam with decoupled weight decay)
    AdamW {
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    },
    /// RMSprop
    RMSprop {
        lr: f64,
        alpha: f64,
        eps: f64,
        weight_decay: f64,
        momentum: f64,
    },
}

impl OptimizerConfig {
    /// Create SGD with default parameters
    pub fn sgd(lr: f64) -> Self {
        OptimizerConfig::SGD {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        }
    }

    /// Create SGD with momentum
    pub fn sgd_momentum(lr: f64, momentum: f64) -> Self {
        OptimizerConfig::SGD {
            lr,
            momentum,
            weight_decay: 0.0,
            nesterov: false,
        }
    }

    /// Create Adam with default parameters
    pub fn adam(lr: f64) -> Self {
        OptimizerConfig::Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }

    /// Create AdamW with default parameters
    pub fn adamw(lr: f64, weight_decay: f64) -> Self {
        OptimizerConfig::AdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
        }
    }
}

/// State for SGD optimizer (momentum buffer)
#[derive(Debug, Clone)]
struct SGDState {
    momentum_buffer: Option<ArrayD<f64>>,
}

/// State for Adam optimizer
#[derive(Debug, Clone)]
struct AdamState {
    /// First moment estimate
    m: ArrayD<f64>,
    /// Second moment estimate
    v: ArrayD<f64>,
    /// Step count for bias correction
    step: usize,
}

/// State for RMSprop optimizer
#[derive(Debug, Clone)]
struct RMSpropState {
    /// Running average of squared gradients
    square_avg: ArrayD<f64>,
    /// Momentum buffer
    momentum_buffer: Option<ArrayD<f64>>,
}

/// Optimizer state union
#[derive(Debug, Clone)]
enum OptimizerState {
    SGD(SGDState),
    Adam(AdamState),
    RMSprop(RMSpropState),
}

/// ML Optimizer for training neural networks
#[derive(Debug)]
pub struct MLOptimizer {
    /// Optimizer configuration
    config: OptimizerConfig,
    /// Per-parameter state
    states: HashMap<ParamId, OptimizerState>,
}

impl MLOptimizer {
    /// Create a new optimizer with the given configuration
    pub fn new(config: OptimizerConfig) -> Self {
        MLOptimizer {
            config,
            states: HashMap::new(),
        }
    }

    /// Get the learning rate
    pub fn learning_rate(&self) -> f64 {
        match &self.config {
            OptimizerConfig::SGD { lr, .. } => *lr,
            OptimizerConfig::Adam { lr, .. } => *lr,
            OptimizerConfig::AdamW { lr, .. } => *lr,
            OptimizerConfig::RMSprop { lr, .. } => *lr,
        }
    }

    /// Set the learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        match &mut self.config {
            OptimizerConfig::SGD { lr: ref mut l, .. } => *l = lr,
            OptimizerConfig::Adam { lr: ref mut l, .. } => *l = lr,
            OptimizerConfig::AdamW { lr: ref mut l, .. } => *l = lr,
            OptimizerConfig::RMSprop { lr: ref mut l, .. } => *l = lr,
        }
    }

    /// Perform one optimization step for all parameters
    pub fn step(
        &mut self,
        params: &mut HashMap<ParamId, TensorValue>,
        grads: &HashMap<ParamId, ArrayD<f64>>,
    ) {
        for (param_id, param) in params.iter_mut() {
            if let Some(grad) = grads.get(param_id) {
                self.step_single(*param_id, param, grad);
            }
        }
    }

    /// Perform optimization step for a single parameter
    fn step_single(&mut self, param_id: ParamId, param: &mut TensorValue, grad: &ArrayD<f64>) {
        match &self.config {
            OptimizerConfig::SGD { lr, momentum, weight_decay, nesterov } => {
                self.sgd_step(param_id, param, grad, *lr, *momentum, *weight_decay, *nesterov);
            }
            OptimizerConfig::Adam { lr, beta1, beta2, eps, weight_decay } => {
                self.adam_step(param_id, param, grad, *lr, *beta1, *beta2, *eps, *weight_decay, false);
            }
            OptimizerConfig::AdamW { lr, beta1, beta2, eps, weight_decay } => {
                self.adam_step(param_id, param, grad, *lr, *beta1, *beta2, *eps, *weight_decay, true);
            }
            OptimizerConfig::RMSprop { lr, alpha, eps, weight_decay, momentum } => {
                self.rmsprop_step(param_id, param, grad, *lr, *alpha, *eps, *weight_decay, *momentum);
            }
        }
    }

    /// SGD optimization step
    fn sgd_step(
        &mut self,
        param_id: ParamId,
        param: &mut TensorValue,
        grad: &ArrayD<f64>,
        lr: f64,
        momentum: f64,
        weight_decay: f64,
        nesterov: bool,
    ) {
        // Apply weight decay (L2 regularization)
        let mut d_p = if weight_decay != 0.0 {
            grad + &(&param.data * weight_decay)
        } else {
            grad.clone()
        };

        // Apply momentum
        if momentum != 0.0 {
            let state = self.states.entry(param_id).or_insert_with(|| {
                OptimizerState::SGD(SGDState { momentum_buffer: None })
            });

            if let OptimizerState::SGD(sgd_state) = state {
                match &mut sgd_state.momentum_buffer {
                    Some(buf) => {
                        *buf = &*buf * momentum + &d_p;
                        d_p = if nesterov {
                            &d_p + &(&*buf * momentum)
                        } else {
                            buf.clone()
                        };
                    }
                    None => {
                        sgd_state.momentum_buffer = Some(d_p.clone());
                    }
                }
            }
        }

        // Update parameter
        param.data = &param.data - &(&d_p * lr);
    }

    /// Adam/AdamW optimization step
    fn adam_step(
        &mut self,
        param_id: ParamId,
        param: &mut TensorValue,
        grad: &ArrayD<f64>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        decoupled_wd: bool, // true for AdamW
    ) {
        // Get or initialize state
        let state = self.states.entry(param_id).or_insert_with(|| {
            OptimizerState::Adam(AdamState {
                m: ArrayD::zeros(param.data.raw_dim()),
                v: ArrayD::zeros(param.data.raw_dim()),
                step: 0,
            })
        });

        if let OptimizerState::Adam(adam_state) = state {
            adam_state.step += 1;
            let step = adam_state.step;

            // Apply L2 regularization to gradient (for Adam, not AdamW)
            let grad_with_wd = if weight_decay != 0.0 && !decoupled_wd {
                grad + &(&param.data * weight_decay)
            } else {
                grad.clone()
            };

            // Update biased first moment estimate
            adam_state.m = &adam_state.m * beta1 + &(&grad_with_wd * (1.0 - beta1));

            // Update biased second raw moment estimate
            adam_state.v = &adam_state.v * beta2 + &(&grad_with_wd * &grad_with_wd * (1.0 - beta2));

            // Compute bias-corrected estimates
            let bias_correction1 = 1.0 - beta1.powi(step as i32);
            let bias_correction2 = 1.0 - beta2.powi(step as i32);

            let m_hat = &adam_state.m / bias_correction1;
            let v_hat = &adam_state.v / bias_correction2;

            // Compute update
            let denom = v_hat.mapv(|x| x.sqrt() + eps);
            let update = &m_hat / &denom * lr;

            // Apply decoupled weight decay (for AdamW)
            if decoupled_wd && weight_decay != 0.0 {
                param.data = &param.data * (1.0 - lr * weight_decay) - &update;
            } else {
                param.data = &param.data - &update;
            }
        }
    }

    /// RMSprop optimization step
    fn rmsprop_step(
        &mut self,
        param_id: ParamId,
        param: &mut TensorValue,
        grad: &ArrayD<f64>,
        lr: f64,
        alpha: f64,
        eps: f64,
        weight_decay: f64,
        momentum: f64,
    ) {
        // Apply weight decay
        let grad_with_wd = if weight_decay != 0.0 {
            grad + &(&param.data * weight_decay)
        } else {
            grad.clone()
        };

        // Get or initialize state
        let state = self.states.entry(param_id).or_insert_with(|| {
            OptimizerState::RMSprop(RMSpropState {
                square_avg: ArrayD::zeros(param.data.raw_dim()),
                momentum_buffer: None,
            })
        });

        if let OptimizerState::RMSprop(rms_state) = state {
            // Update running average of squared gradients
            rms_state.square_avg = &rms_state.square_avg * alpha
                + &(&grad_with_wd * &grad_with_wd * (1.0 - alpha));

            // Compute update
            let avg = rms_state.square_avg.mapv(|x| x.sqrt() + eps);
            let mut update = &grad_with_wd / &avg;

            // Apply momentum
            if momentum != 0.0 {
                match &mut rms_state.momentum_buffer {
                    Some(buf) => {
                        *buf = &*buf * momentum + &update;
                        update = buf.clone();
                    }
                    None => {
                        rms_state.momentum_buffer = Some(update.clone());
                    }
                }
            }

            // Update parameter
            param.data = &param.data - &(&update * lr);
        }
    }

    /// Zero all gradients
    pub fn zero_grad(&mut self, params: &mut HashMap<ParamId, TensorValue>) {
        for param in params.values_mut() {
            param.zero_grad();
        }
    }

    /// Reset optimizer state (useful when restarting training)
    pub fn reset_state(&mut self) {
        self.states.clear();
    }

    /// Get the number of parameters being optimized
    pub fn num_params(&self) -> usize {
        self.states.len()
    }
}

/// Learning rate scheduler
#[derive(Debug, Clone)]
pub enum LRScheduler {
    /// Constant learning rate
    Constant,
    /// Step decay: lr = lr * gamma every step_size epochs
    StepLR { step_size: usize, gamma: f64 },
    /// Exponential decay: lr = lr * gamma^epoch
    ExponentialLR { gamma: f64 },
    /// Cosine annealing: lr varies between lr_min and lr_max
    CosineAnnealing { t_max: usize, lr_min: f64 },
    /// Linear warmup followed by constant
    LinearWarmup { warmup_steps: usize, target_lr: f64 },
}

impl LRScheduler {
    /// Compute learning rate for given epoch/step
    pub fn get_lr(&self, base_lr: f64, epoch: usize, step: usize) -> f64 {
        match self {
            LRScheduler::Constant => base_lr,

            LRScheduler::StepLR { step_size, gamma } => {
                let num_decays = epoch / step_size;
                base_lr * gamma.powi(num_decays as i32)
            }

            LRScheduler::ExponentialLR { gamma } => {
                base_lr * gamma.powi(epoch as i32)
            }

            LRScheduler::CosineAnnealing { t_max, lr_min } => {
                let progress = (epoch % t_max) as f64 / *t_max as f64;
                lr_min + (base_lr - lr_min) * (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0
            }

            LRScheduler::LinearWarmup { warmup_steps, target_lr } => {
                if step < *warmup_steps {
                    *target_lr * (step as f64 / *warmup_steps as f64)
                } else {
                    *target_lr
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_step() {
        let mut optimizer = MLOptimizer::new(OptimizerConfig::sgd(0.1));

        let mut params: HashMap<ParamId, TensorValue> = HashMap::new();
        params.insert(0, TensorValue::vector(vec![1.0, 2.0, 3.0]));

        let mut grads: HashMap<ParamId, ArrayD<f64>> = HashMap::new();
        grads.insert(0, ndarray::arr1(&[0.1, 0.2, 0.3]).into_dyn());

        optimizer.step(&mut params, &grads);

        // Check that parameters were updated
        let param = params.get(&0).unwrap();
        assert!((param.data[[0]] - 0.99).abs() < 1e-6); // 1.0 - 0.1 * 0.1
        assert!((param.data[[1]] - 1.98).abs() < 1e-6); // 2.0 - 0.1 * 0.2
    }

    #[test]
    fn test_adam_step() {
        let mut optimizer = MLOptimizer::new(OptimizerConfig::adam(0.001));

        let mut params: HashMap<ParamId, TensorValue> = HashMap::new();
        params.insert(0, TensorValue::vector(vec![1.0, 2.0, 3.0]));

        let mut grads: HashMap<ParamId, ArrayD<f64>> = HashMap::new();
        grads.insert(0, ndarray::arr1(&[0.1, 0.2, 0.3]).into_dyn());

        // Run multiple steps
        for _ in 0..10 {
            optimizer.step(&mut params, &grads);
        }

        // Parameters should have moved toward zero (gradient descent on positive gradients)
        let param = params.get(&0).unwrap();
        assert!(param.data[[0]] < 1.0);
    }

    #[test]
    fn test_lr_scheduler() {
        let scheduler = LRScheduler::StepLR { step_size: 10, gamma: 0.1 };

        assert!((scheduler.get_lr(0.1, 0, 0) - 0.1).abs() < 1e-6);
        assert!((scheduler.get_lr(0.1, 10, 0) - 0.01).abs() < 1e-6);
        assert!((scheduler.get_lr(0.1, 20, 0) - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_warmup_scheduler() {
        let scheduler = LRScheduler::LinearWarmup { warmup_steps: 100, target_lr: 0.001 };

        assert!((scheduler.get_lr(0.0, 0, 0) - 0.0).abs() < 1e-6);
        assert!((scheduler.get_lr(0.0, 0, 50) - 0.0005).abs() < 1e-6);
        assert!((scheduler.get_lr(0.0, 0, 100) - 0.001).abs() < 1e-6);
        assert!((scheduler.get_lr(0.0, 0, 200) - 0.001).abs() < 1e-6);
    }
}
