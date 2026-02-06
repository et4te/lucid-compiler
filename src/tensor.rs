//! Tensor values and operations for ML training support
//!
//! This module provides multi-dimensional array support with automatic
//! differentiation capabilities for training neural networks.

use ndarray::{Array, ArrayD, IxDyn, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Uniform, Normal};
use std::fmt;
use std::ops::{Add, Sub, Mul, Div};

/// A tensor value with optional gradient tracking for automatic differentiation
#[derive(Clone)]
pub struct TensorValue {
    /// The tensor data
    pub data: ArrayD<f64>,
    /// Whether this tensor requires gradient computation
    pub requires_grad: bool,
    /// Accumulated gradient (set during backward pass)
    pub grad: Option<ArrayD<f64>>,
}

impl TensorValue {
    /// Create a new tensor from an ndarray
    pub fn new(data: ArrayD<f64>) -> Self {
        TensorValue {
            data,
            requires_grad: false,
            grad: None,
        }
    }

    /// Create a tensor that requires gradient
    pub fn with_grad(data: ArrayD<f64>) -> Self {
        TensorValue {
            data,
            requires_grad: true,
            grad: None,
        }
    }

    /// Create a scalar tensor
    pub fn scalar(value: f64) -> Self {
        TensorValue::new(Array::from_elem(IxDyn(&[]), value))
    }

    /// Create a 1D tensor (vector)
    pub fn vector(values: Vec<f64>) -> Self {
        let len = values.len();
        TensorValue::new(Array::from_shape_vec(IxDyn(&[len]), values).unwrap())
    }

    /// Create a 2D tensor (matrix)
    pub fn matrix(values: Vec<Vec<f64>>) -> Self {
        let rows = values.len();
        let cols = if rows > 0 { values[0].len() } else { 0 };
        let flat: Vec<f64> = values.into_iter().flatten().collect();
        TensorValue::new(Array::from_shape_vec(IxDyn(&[rows, cols]), flat).unwrap())
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: &[usize]) -> Self {
        TensorValue::new(Array::zeros(IxDyn(shape)))
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: &[usize]) -> Self {
        TensorValue::new(Array::ones(IxDyn(shape)))
    }

    /// Create a tensor with random uniform values
    pub fn rand_uniform(shape: &[usize], low: f64, high: f64) -> Self {
        TensorValue::new(Array::random(IxDyn(shape), Uniform::new(low, high)))
    }

    /// Create a tensor with random normal values
    pub fn rand_normal(shape: &[usize], mean: f64, std: f64) -> Self {
        TensorValue::new(Array::random(IxDyn(shape), Normal::new(mean, std).unwrap()))
    }

    /// Xavier/Glorot initialization for weights
    pub fn xavier(shape: &[usize]) -> Self {
        let fan_in = if shape.len() >= 2 { shape[shape.len() - 2] } else { 1 };
        let fan_out = if shape.len() >= 1 { shape[shape.len() - 1] } else { 1 };
        let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
        TensorValue::with_grad(Array::random(IxDyn(shape), Normal::new(0.0, std).unwrap()))
    }

    /// He initialization for ReLU networks
    pub fn he(shape: &[usize]) -> Self {
        let fan_in = if shape.len() >= 2 { shape[shape.len() - 2] } else { 1 };
        let std = (2.0 / fan_in as f64).sqrt();
        TensorValue::with_grad(Array::random(IxDyn(shape), Normal::new(0.0, std).unwrap()))
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Check if tensor is a scalar
    pub fn is_scalar(&self) -> bool {
        self.data.ndim() == 0
    }

    /// Get scalar value (panics if not scalar)
    pub fn to_scalar(&self) -> f64 {
        assert!(self.is_scalar(), "Tensor is not a scalar");
        self.data[[]]
    }

    /// Zero out the gradient
    pub fn zero_grad(&mut self) {
        if self.requires_grad {
            self.grad = Some(Array::zeros(self.data.raw_dim()));
        }
    }

    /// Accumulate gradient
    pub fn accumulate_grad(&mut self, grad: &ArrayD<f64>) {
        if self.requires_grad {
            match &mut self.grad {
                Some(existing) => *existing = existing.clone() + grad,
                None => self.grad = Some(grad.clone()),
            }
        }
    }

    // ============ Tensor Operations ============

    /// Element-wise addition
    pub fn add(&self, other: &TensorValue) -> TensorValue {
        TensorValue::new(&self.data + &other.data)
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &TensorValue) -> TensorValue {
        TensorValue::new(&self.data - &other.data)
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &TensorValue) -> TensorValue {
        TensorValue::new(&self.data * &other.data)
    }

    /// Element-wise division
    pub fn div(&self, other: &TensorValue) -> TensorValue {
        TensorValue::new(&self.data / &other.data)
    }

    /// Scalar addition
    pub fn add_scalar(&self, scalar: f64) -> TensorValue {
        TensorValue::new(&self.data + scalar)
    }

    /// Scalar multiplication
    pub fn mul_scalar(&self, scalar: f64) -> TensorValue {
        TensorValue::new(&self.data * scalar)
    }

    /// Matrix multiplication (2D tensors)
    pub fn matmul(&self, other: &TensorValue) -> TensorValue {
        assert!(self.ndim() == 2 && other.ndim() == 2, "matmul requires 2D tensors");

        let a = self.data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b = other.data.view().into_dimensionality::<ndarray::Ix2>().unwrap();

        let result = a.dot(&b);
        TensorValue::new(result.into_dyn())
    }

    /// Batched matrix multiplication for 3D tensors [batch, m, n] @ [batch, n, p]
    pub fn bmm(&self, other: &TensorValue) -> TensorValue {
        assert!(self.ndim() == 3 && other.ndim() == 3, "bmm requires 3D tensors");

        let batch_size = self.shape()[0];
        assert_eq!(batch_size, other.shape()[0], "Batch sizes must match");

        let m = self.shape()[1];
        let p = other.shape()[2];

        let mut result = Array::zeros(IxDyn(&[batch_size, m, p]));

        for b in 0..batch_size {
            let a_slice = self.data.slice(s![b, .., ..]);
            let b_slice = other.data.slice(s![b, .., ..]);
            let a_2d = a_slice.into_dimensionality::<ndarray::Ix2>().unwrap();
            let b_2d = b_slice.into_dimensionality::<ndarray::Ix2>().unwrap();
            let prod = a_2d.dot(&b_2d);
            result.slice_mut(s![b, .., ..]).assign(&prod);
        }

        TensorValue::new(result)
    }

    /// Transpose (swap last two dimensions)
    pub fn transpose(&self) -> TensorValue {
        if self.ndim() < 2 {
            return self.clone();
        }

        let mut axes: Vec<usize> = (0..self.ndim()).collect();
        let n = axes.len();
        axes.swap(n - 1, n - 2);

        TensorValue::new(self.data.clone().permuted_axes(IxDyn(&axes)))
    }

    /// Reshape tensor to new shape
    pub fn reshape(&self, new_shape: &[usize]) -> TensorValue {
        let total: usize = new_shape.iter().product();
        assert_eq!(total, self.len(), "New shape must have same total elements");
        TensorValue::new(self.data.clone().into_shape(IxDyn(new_shape)).unwrap())
    }

    /// Sum all elements
    pub fn sum(&self) -> TensorValue {
        TensorValue::scalar(self.data.sum())
    }

    /// Sum along an axis
    pub fn sum_axis(&self, axis: usize) -> TensorValue {
        TensorValue::new(self.data.sum_axis(Axis(axis)))
    }

    /// Mean of all elements
    pub fn mean(&self) -> TensorValue {
        TensorValue::scalar(self.data.mean().unwrap_or(0.0))
    }

    /// Mean along an axis
    pub fn mean_axis(&self, axis: usize) -> TensorValue {
        TensorValue::new(self.data.mean_axis(Axis(axis)).unwrap())
    }

    /// Element-wise maximum with zero (ReLU)
    pub fn relu(&self) -> TensorValue {
        TensorValue::new(self.data.mapv(|x| x.max(0.0)))
    }

    /// Element-wise sigmoid
    pub fn sigmoid(&self) -> TensorValue {
        TensorValue::new(self.data.mapv(|x| 1.0 / (1.0 + (-x).exp())))
    }

    /// Element-wise tanh
    pub fn tanh(&self) -> TensorValue {
        TensorValue::new(self.data.mapv(|x| x.tanh()))
    }

    /// Element-wise exponential
    pub fn exp(&self) -> TensorValue {
        TensorValue::new(self.data.mapv(|x| x.exp()))
    }

    /// Element-wise natural log
    pub fn log(&self) -> TensorValue {
        TensorValue::new(self.data.mapv(|x| x.ln()))
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> TensorValue {
        TensorValue::new(self.data.mapv(|x| x.sqrt()))
    }

    /// Element-wise power
    pub fn pow(&self, exp: f64) -> TensorValue {
        TensorValue::new(self.data.mapv(|x| x.powf(exp)))
    }

    /// Softmax along last axis
    pub fn softmax(&self) -> TensorValue {
        let last_axis = self.ndim() - 1;

        // Subtract max for numerical stability
        let max_vals = self.data.map_axis(Axis(last_axis), |row| {
            row.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        });

        // Expand max_vals to broadcast
        let max_expanded = max_vals.insert_axis(Axis(last_axis));

        let exp_vals = (&self.data - &max_expanded).mapv(|x| x.exp());
        let sum_exp = exp_vals.sum_axis(Axis(last_axis)).insert_axis(Axis(last_axis));

        TensorValue::new(exp_vals / sum_exp)
    }

    /// Layer normalization along last axis
    pub fn layer_norm(&self, eps: f64) -> TensorValue {
        let last_axis = self.ndim() - 1;
        let mean = self.data.mean_axis(Axis(last_axis)).unwrap().insert_axis(Axis(last_axis));
        let centered = &self.data - &mean;
        let var = centered.mapv(|x| x * x).mean_axis(Axis(last_axis)).unwrap().insert_axis(Axis(last_axis));
        let std = var.mapv(|x| (x + eps).sqrt());
        TensorValue::new(centered / std)
    }

    /// Dropout (for training)
    pub fn dropout(&self, p: f64, training: bool) -> TensorValue {
        if !training || p == 0.0 {
            return self.clone();
        }

        let mask = Array::random(self.data.raw_dim(), Uniform::new(0.0, 1.0))
            .mapv(|x| if x > p { 1.0 / (1.0 - p) } else { 0.0 });

        TensorValue::new(&self.data * &mask)
    }

    /// Scale by a scalar value
    pub fn scale(&self, scalar: f64) -> TensorValue {
        TensorValue::new(&self.data * scalar)
    }

    /// Add bias with broadcasting
    /// Assumes bias has shape [hidden_dim] and self has shape [..., hidden_dim]
    pub fn add_bias(&self, bias: &TensorValue) -> TensorValue {
        let bias_expanded = bias.data.broadcast(self.data.raw_dim()).unwrap();
        TensorValue::new(&self.data + &bias_expanded)
    }

    /// Concatenate tensors along a given axis
    pub fn concat(&self, other: &TensorValue, axis: usize) -> TensorValue {
        use ndarray::concatenate;
        let result = concatenate(
            Axis(axis),
            &[self.data.view(), other.data.view()]
        ).unwrap();
        TensorValue::new(result)
    }

    /// Layer normalization with learned parameters
    pub fn layer_norm_with_params(&self, weight: &TensorValue, bias: &TensorValue) -> TensorValue {
        let eps = 1e-5;
        let normalized = self.layer_norm(eps);

        // Apply scale and shift: weight * normalized + bias
        let scaled = normalized.mul(weight);
        scaled.add(bias)
    }

    /// Negation
    pub fn neg(&self) -> TensorValue {
        TensorValue::new(-&self.data)
    }

    /// Element-wise clamp
    pub fn clamp(&self, min: f64, max: f64) -> TensorValue {
        TensorValue::new(self.data.mapv(|x| x.max(min).min(max)))
    }

    /// Get a slice along an axis
    pub fn slice_axis(&self, axis: usize, start: usize, end: usize) -> TensorValue {
        let mut slices = vec![ndarray::SliceInfoElem::Slice {
            start: 0,
            end: None,
            step: 1,
        }; self.ndim()];
        slices[axis] = ndarray::SliceInfoElem::Slice {
            start: start as isize,
            end: Some(end as isize),
            step: 1,
        };
        let slice_info = ndarray::SliceInfo::<_, IxDyn, IxDyn>::try_from(slices).unwrap();
        TensorValue::new(self.data.slice(slice_info).to_owned())
    }

    /// Argmax along last axis
    pub fn argmax(&self) -> Vec<usize> {
        if self.ndim() == 1 {
            let idx = self.data.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            vec![idx]
        } else {
            let last_axis = self.ndim() - 1;
            self.data.map_axis(Axis(last_axis), |row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }).iter().cloned().collect()
        }
    }
}

impl fmt::Debug for TensorValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TensorValue(shape={:?}, requires_grad={})", self.shape(), self.requires_grad)
    }
}

impl fmt::Display for TensorValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_scalar() {
            write!(f, "{:.4}", self.to_scalar())
        } else {
            write!(f, "Tensor{:?}", self.shape())
        }
    }
}

impl PartialEq for TensorValue {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

// Operator overloading for convenience
impl Add for &TensorValue {
    type Output = TensorValue;
    fn add(self, other: &TensorValue) -> TensorValue {
        self.add(other)
    }
}

impl Sub for &TensorValue {
    type Output = TensorValue;
    fn sub(self, other: &TensorValue) -> TensorValue {
        TensorValue::sub(self, other)
    }
}

impl Mul for &TensorValue {
    type Output = TensorValue;
    fn mul(self, other: &TensorValue) -> TensorValue {
        TensorValue::mul(self, other)
    }
}

impl Div for &TensorValue {
    type Output = TensorValue;
    fn div(self, other: &TensorValue) -> TensorValue {
        TensorValue::div(self, other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let scalar = TensorValue::scalar(5.0);
        assert!(scalar.is_scalar());
        assert_eq!(scalar.to_scalar(), 5.0);

        let vec = TensorValue::vector(vec![1.0, 2.0, 3.0]);
        assert_eq!(vec.shape(), vec![3]);

        let mat = TensorValue::matrix(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        assert_eq!(mat.shape(), vec![2, 2]);
    }

    #[test]
    fn test_tensor_ops() {
        let a = TensorValue::vector(vec![1.0, 2.0, 3.0]);
        let b = TensorValue::vector(vec![4.0, 5.0, 6.0]);

        let sum = a.add(&b);
        assert_eq!(sum.data[[0]], 5.0);
        assert_eq!(sum.data[[1]], 7.0);
        assert_eq!(sum.data[[2]], 9.0);
    }

    #[test]
    fn test_matmul() {
        let a = TensorValue::matrix(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let b = TensorValue::matrix(vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ]);

        let c = a.matmul(&b);
        assert_eq!(c.shape(), vec![2, 2]);
        assert_eq!(c.data[[0, 0]], 19.0); // 1*5 + 2*7
        assert_eq!(c.data[[0, 1]], 22.0); // 1*6 + 2*8
    }

    #[test]
    fn test_softmax() {
        let x = TensorValue::vector(vec![1.0, 2.0, 3.0]);
        let s = x.softmax();

        // Softmax should sum to 1
        let sum: f64 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_relu() {
        let x = TensorValue::vector(vec![-1.0, 0.0, 1.0, 2.0]);
        let r = x.relu();

        assert_eq!(r.data[[0]], 0.0);
        assert_eq!(r.data[[1]], 0.0);
        assert_eq!(r.data[[2]], 1.0);
        assert_eq!(r.data[[3]], 2.0);
    }

    #[test]
    fn test_xavier_init() {
        let w = TensorValue::xavier(&[100, 50]);
        assert_eq!(w.shape(), vec![100, 50]);
        assert!(w.requires_grad);

        // Check that values are roughly in expected range (should be close to 0 mean)
        let mean: f64 = w.data.mean().unwrap();
        assert!(mean.abs() < 0.1);
    }
}
