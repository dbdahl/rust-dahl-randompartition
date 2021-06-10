use crate::clust::Clustering;
use crate::distr::{PartitionSampler, ProbabilityMassFunction};

use rand::prelude::*;
use std::convert::TryFrom;

pub fn sample_into_slice<S: PartitionSampler, T: Rng, F: Fn(&mut S, &mut T)>(
    n_partitions: usize,
    n_items: usize,
    matrix: &mut [i32],
    rng: &mut T,
    distr: &mut S,
    callback: F,
) {
    for i in 0..n_partitions {
        callback(distr, rng);
        let p = distr.sample(rng).standardize();
        let labels = p.allocation();
        for j in 0..n_items {
            matrix[n_partitions * j + i] = i32::try_from(labels[j] + 1).unwrap();
        }
    }
}

pub fn log_probabilities_into_slice<S: ProbabilityMassFunction>(
    n_partitions: usize,
    n_items: usize,
    matrix: &[i32],
    log_probabilities: &mut [f64],
    distr: &mut S,
) {
    for i in 0..n_partitions {
        let mut target_labels = Vec::with_capacity(n_items);
        for j in 0..n_items {
            target_labels.push(matrix[n_partitions * j + i] as usize);
        }
        let target = Clustering::from_vector(target_labels);
        log_probabilities[i] = distr.log_pmf(&target);
    }
}
