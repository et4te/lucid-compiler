//! Data Loading Utilities for ML Training
//!
//! This module provides utilities for loading and batching training data.

use crate::tensor::TensorValue;
use ndarray::{Array, IxDyn};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// A single training sample
#[derive(Debug, Clone)]
pub struct Sample {
    pub input: TensorValue,
    pub target: TensorValue,
}

/// Dataset holding training samples
#[derive(Debug, Clone)]
pub struct Dataset {
    samples: Vec<Sample>,
}

impl Dataset {
    /// Create a new empty dataset
    pub fn new() -> Self {
        Dataset { samples: Vec::new() }
    }

    /// Create a dataset from samples
    pub fn from_samples(samples: Vec<Sample>) -> Self {
        Dataset { samples }
    }

    /// Add a sample to the dataset
    pub fn push(&mut self, sample: Sample) {
        self.samples.push(sample);
    }

    /// Get the number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get a sample by index
    pub fn get(&self, index: usize) -> Option<&Sample> {
        self.samples.get(index)
    }

    /// Shuffle the dataset
    pub fn shuffle(&mut self) {
        let mut rng = thread_rng();
        self.samples.shuffle(&mut rng);
    }

    /// Split dataset into train and validation sets
    pub fn split(&self, train_ratio: f64) -> (Dataset, Dataset) {
        let split_idx = (self.samples.len() as f64 * train_ratio) as usize;
        let train = Dataset::from_samples(self.samples[..split_idx].to_vec());
        let val = Dataset::from_samples(self.samples[split_idx..].to_vec());
        (train, val)
    }

    /// Create a synthetic dataset for testing
    pub fn synthetic_classification(num_samples: usize, input_dim: usize, num_classes: usize) -> Self {
        let mut samples = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            // Generate random input
            let input = TensorValue::rand_normal(&[input_dim], 0.0, 1.0);

            // Generate one-hot target (cycle through classes)
            let class_idx = i % num_classes;
            let mut target_data = vec![0.0; num_classes];
            target_data[class_idx] = 1.0;
            let target = TensorValue::vector(target_data);

            samples.push(Sample { input, target });
        }

        Dataset { samples }
    }

    /// Create a synthetic dataset for sequence modeling
    pub fn synthetic_sequence(num_samples: usize, seq_len: usize, vocab_size: usize) -> Self {
        let mut samples = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            // Generate random token IDs (as floats for now)
            let input_data: Vec<f64> = (0..seq_len)
                .map(|_| (rand::random::<f64>() * vocab_size as f64).floor())
                .collect();
            let input = TensorValue::vector(input_data.clone());

            // Target is shifted input (next token prediction)
            let mut target_data = input_data[1..].to_vec();
            target_data.push((rand::random::<f64>() * vocab_size as f64).floor());
            let target = TensorValue::vector(target_data);

            samples.push(Sample { input, target });
        }

        Dataset { samples }
    }

    /// Load dataset from CSV file
    /// Format: each row is input1,input2,...,target1,target2,...
    pub fn from_csv(path: &str, input_cols: usize, target_cols: usize) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut samples = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let values: Vec<f64> = line
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();

            if values.len() >= input_cols + target_cols {
                let input = TensorValue::vector(values[..input_cols].to_vec());
                let target = TensorValue::vector(values[input_cols..input_cols + target_cols].to_vec());
                samples.push(Sample { input, target });
            }
        }

        Ok(Dataset { samples })
    }
}

impl Default for Dataset {
    fn default() -> Self {
        Self::new()
    }
}

/// DataLoader for batching and iterating over datasets
#[derive(Debug)]
pub struct DataLoader {
    dataset: Dataset,
    batch_size: usize,
    shuffle: bool,
    current_idx: usize,
    indices: Vec<usize>,
}

impl DataLoader {
    /// Create a new DataLoader
    pub fn new(dataset: Dataset, batch_size: usize, shuffle: bool) -> Self {
        let indices: Vec<usize> = (0..dataset.len()).collect();
        DataLoader {
            dataset,
            batch_size,
            shuffle,
            current_idx: 0,
            indices,
        }
    }

    /// Get the number of batches
    pub fn num_batches(&self) -> usize {
        (self.dataset.len() + self.batch_size - 1) / self.batch_size
    }

    /// Reset the loader for a new epoch
    pub fn reset(&mut self) {
        self.current_idx = 0;
        if self.shuffle {
            let mut rng = thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }

    /// Get the next batch
    pub fn next_batch(&mut self) -> Option<(TensorValue, TensorValue)> {
        if self.current_idx >= self.dataset.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.dataset.len());
        let batch_indices = &self.indices[self.current_idx..end_idx];

        // Collect batch samples
        let batch_samples: Vec<&Sample> = batch_indices
            .iter()
            .map(|&i| self.dataset.get(i).unwrap())
            .collect();

        // Stack inputs and targets
        let inputs = Self::stack_tensors(
            batch_samples.iter().map(|s| &s.input).collect()
        );
        let targets = Self::stack_tensors(
            batch_samples.iter().map(|s| &s.target).collect()
        );

        self.current_idx = end_idx;

        Some((inputs, targets))
    }

    /// Stack multiple tensors into a batch (adds batch dimension)
    fn stack_tensors(tensors: Vec<&TensorValue>) -> TensorValue {
        if tensors.is_empty() {
            return TensorValue::zeros(&[0]);
        }

        let batch_size = tensors.len();
        let sample_shape = tensors[0].shape();

        // New shape: [batch_size, ...sample_shape]
        let mut new_shape = vec![batch_size];
        new_shape.extend(sample_shape.iter());

        // Flatten all data
        let all_data: Vec<f64> = tensors
            .iter()
            .flat_map(|t| t.data.iter().cloned())
            .collect();

        TensorValue::new(
            Array::from_shape_vec(IxDyn(&new_shape), all_data).unwrap()
        )
    }
}

impl Iterator for DataLoader {
    type Item = (TensorValue, TensorValue);

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}

/// Batch collation utilities
pub mod collate {
    use super::*;

    /// Pad sequences to the same length
    pub fn pad_sequences(sequences: &[TensorValue], pad_value: f64) -> TensorValue {
        if sequences.is_empty() {
            return TensorValue::zeros(&[0]);
        }

        let batch_size = sequences.len();
        let max_len = sequences.iter().map(|s| s.shape()[0]).max().unwrap_or(0);

        let mut padded_data = vec![pad_value; batch_size * max_len];

        for (i, seq) in sequences.iter().enumerate() {
            let seq_len = seq.shape()[0];
            for j in 0..seq_len {
                padded_data[i * max_len + j] = seq.data[[j]];
            }
        }

        TensorValue::new(
            Array::from_shape_vec(IxDyn(&[batch_size, max_len]), padded_data).unwrap()
        )
    }

    /// Create attention mask for padded sequences
    pub fn create_attention_mask(sequences: &[TensorValue], _pad_value: f64) -> TensorValue {
        if sequences.is_empty() {
            return TensorValue::zeros(&[0]);
        }

        let batch_size = sequences.len();
        let max_len = sequences.iter().map(|s| s.shape()[0]).max().unwrap_or(0);

        let mut mask_data = vec![0.0; batch_size * max_len];

        for (i, seq) in sequences.iter().enumerate() {
            let seq_len = seq.shape()[0];
            for j in 0..seq_len {
                mask_data[i * max_len + j] = 1.0;
            }
        }

        TensorValue::new(
            Array::from_shape_vec(IxDyn(&[batch_size, max_len]), mask_data).unwrap()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_creation() {
        let dataset = Dataset::synthetic_classification(100, 10, 5);
        assert_eq!(dataset.len(), 100);
    }

    #[test]
    fn test_dataset_split() {
        let dataset = Dataset::synthetic_classification(100, 10, 5);
        let (train, val) = dataset.split(0.8);
        assert_eq!(train.len(), 80);
        assert_eq!(val.len(), 20);
    }

    #[test]
    fn test_dataloader_batching() {
        let dataset = Dataset::synthetic_classification(100, 10, 5);
        let mut loader = DataLoader::new(dataset, 32, false);

        let mut total_samples = 0;
        while let Some((inputs, targets)) = loader.next_batch() {
            total_samples += inputs.shape()[0];
            assert!(inputs.shape()[0] <= 32);
            assert_eq!(inputs.shape()[1], 10);
            assert_eq!(targets.shape()[1], 5);
        }

        assert_eq!(total_samples, 100);
    }

    #[test]
    fn test_dataloader_shuffle() {
        let dataset = Dataset::synthetic_classification(10, 5, 2);
        let mut loader = DataLoader::new(dataset, 10, true);

        let (batch1, _) = loader.next_batch().unwrap();
        loader.reset();
        let (batch2, _) = loader.next_batch().unwrap();

        // After shuffle, order should likely be different
        // (small chance of being the same, but statistically unlikely)
        // Just verify we get valid batches
        assert_eq!(batch1.shape()[0], 10);
        assert_eq!(batch2.shape()[0], 10);
    }

    #[test]
    fn test_pad_sequences() {
        let seq1 = TensorValue::vector(vec![1.0, 2.0, 3.0]);
        let seq2 = TensorValue::vector(vec![4.0, 5.0]);
        let seq3 = TensorValue::vector(vec![6.0]);

        let padded = collate::pad_sequences(&[seq1, seq2, seq3], 0.0);

        assert_eq!(padded.shape(), vec![3, 3]);
        assert_eq!(padded.data[[0, 0]], 1.0);
        assert_eq!(padded.data[[1, 0]], 4.0);
        assert_eq!(padded.data[[1, 2]], 0.0); // Padded
        assert_eq!(padded.data[[2, 1]], 0.0); // Padded
    }
}
