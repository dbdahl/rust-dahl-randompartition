use crate::clust::Clustering;
use crate::cpp::CPPParameters;
use crate::crp::CRPParameters;
use crate::epa::EPAParameters;
use crate::fixed::FixedPartitionParameters;
use crate::frp::FRPParameters;
use crate::lsp::LSPParameters;
use crate::trp::TRPParameters;
use crate::urp::URPParameters;
use dahl_roxido::mk_rng_isaac;
use rand::prelude::*;
use rand_isaac::IsaacRng;
use std::convert::TryFrom;
use std::ffi::c_void;
use std::slice;

pub trait PartitionSampler {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering;
}

pub trait PartitionLogProbability {
    fn log_probability(&self, partition: &Clustering) -> f64;
    fn is_normalized(&self) -> bool;
}

pub fn sample_into_slice<S: PartitionSampler, T: Rng, F: Fn(&mut S, &mut T) -> ()>(
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

pub fn sample_into_slice2<
    S: crate::distr::PartitionSampler,
    T: Rng,
    F: Fn(&mut S, &mut T) -> (),
>(
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

pub fn log_probabilities_into_slice<S: PartitionLogProbability>(
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
        log_probabilities[i] = distr.log_probability(&target);
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__sample_partition(
    n_partitions: i32,
    n_items: i32,
    partition_labels_ptr: *mut i32,
    seed_ptr: *const i32, // Assumed length is 32
    prior_id: i32,
    prior_ptr: *const c_void,
    randomize_permutation: bool,
) -> () {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let matrix: &mut [i32] = slice::from_raw_parts_mut(partition_labels_ptr, np * ni);
    let rng = &mut mk_rng_isaac(seed_ptr);
    match prior_id {
        0 => {
            let mut p = std::ptr::NonNull::new(prior_ptr as *mut FixedPartitionParameters).unwrap();
            let callback = |_p: &mut FixedPartitionParameters, _rng: &mut IsaacRng| {};
            sample_into_slice(np, ni, matrix, rng, p.as_mut(), callback);
        }
        1 => {
            let mut p = std::ptr::NonNull::new(prior_ptr as *mut CRPParameters).unwrap();
            let callback = |_p: &mut CRPParameters, _rng: &mut IsaacRng| {};
            sample_into_slice(np, ni, matrix, rng, p.as_mut(), callback);
        }
        2 => {
            let mut p = std::ptr::NonNull::new(prior_ptr as *mut FRPParameters).unwrap();
            if randomize_permutation {
                let callback = |p: &mut FRPParameters, rng: &mut IsaacRng| {
                    p.shuffle_permutation(rng);
                };
                sample_into_slice(np, ni, matrix, rng, p.as_mut(), callback);
            } else {
                let callback = |_p: &mut FRPParameters, _rng: &mut IsaacRng| {};
                sample_into_slice(np, ni, matrix, rng, p.as_mut(), callback);
            }
        }
        3 => {
            let mut p = std::ptr::NonNull::new(prior_ptr as *mut LSPParameters).unwrap();
            if randomize_permutation {
                let callback = |p: &mut LSPParameters, rng: &mut IsaacRng| {
                    p.shuffle_permutation(rng);
                };
                sample_into_slice(np, ni, matrix, rng, p.as_mut(), callback);
            } else {
                let callback = |_p: &mut LSPParameters, _rng: &mut IsaacRng| {};
                sample_into_slice(np, ni, matrix, rng, p.as_mut(), callback);
            }
        }
        4 => {
            panic!("Cannot sample from the CPP ({})", prior_id);
        }
        5 => {
            let mut p = std::ptr::NonNull::new(prior_ptr as *mut EPAParameters).unwrap();
            if randomize_permutation {
                let callback = |p: &mut EPAParameters, rng: &mut IsaacRng| {
                    p.shuffle_permutation(rng);
                };
                sample_into_slice(np, ni, matrix, rng, p.as_mut(), callback);
            } else {
                let callback = |_p: &mut EPAParameters, _rng: &mut IsaacRng| {};
                sample_into_slice(np, ni, matrix, rng, p.as_mut(), callback);
            }
        }
        6 => {
            let mut p = std::ptr::NonNull::new(prior_ptr as *mut TRPParameters).unwrap();
            if randomize_permutation {
                let callback = |p: &mut TRPParameters, rng: &mut IsaacRng| {
                    p.shuffle_permutation(rng);
                };
                sample_into_slice(np, ni, matrix, rng, p.as_mut(), callback);
            } else {
                let callback = |_p: &mut TRPParameters, _rng: &mut IsaacRng| {};
                sample_into_slice(np, ni, matrix, rng, p.as_mut(), callback);
            }
        }
        7 => {
            let mut p = std::ptr::NonNull::new(prior_ptr as *mut URPParameters).unwrap();
            let callback = |_p: &mut URPParameters, _rng: &mut IsaacRng| {};
            sample_into_slice2(np, ni, matrix, rng, p.as_mut(), callback);
        }
        _ => panic!("Unsupported prior ID: {}", prior_id),
    };
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__log_probability_of_partition(
    n_partitions: i32,
    n_items: i32,
    partition_labels_ptr: *mut i32,
    log_probabilities_ptr: *mut f64,
    prior_id: i32,
    prior_ptr: *const c_void,
) -> () {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let matrix: &[i32] = slice::from_raw_parts(partition_labels_ptr, np * ni);
    let log_probabilities: &mut [f64] = slice::from_raw_parts_mut(log_probabilities_ptr, np);
    match prior_id {
        0 => {
            let mut p = std::ptr::NonNull::new(prior_ptr as *mut FixedPartitionParameters).unwrap();
            log_probabilities_into_slice(np, ni, matrix, log_probabilities, p.as_mut());
        }
        1 => {
            let mut p = std::ptr::NonNull::new(prior_ptr as *mut CRPParameters).unwrap();
            log_probabilities_into_slice(np, ni, matrix, log_probabilities, p.as_mut());
        }
        2 => {
            let mut p = std::ptr::NonNull::new(prior_ptr as *mut FRPParameters).unwrap();
            log_probabilities_into_slice(np, ni, matrix, log_probabilities, p.as_mut());
        }
        3 => {
            let mut p = std::ptr::NonNull::new(prior_ptr as *mut LSPParameters).unwrap();
            log_probabilities_into_slice(np, ni, matrix, log_probabilities, p.as_mut());
        }
        4 => {
            let mut p = std::ptr::NonNull::new(prior_ptr as *mut CPPParameters).unwrap();
            log_probabilities_into_slice(np, ni, matrix, log_probabilities, p.as_mut());
        }
        5 => {
            let mut p = std::ptr::NonNull::new(prior_ptr as *mut EPAParameters).unwrap();
            log_probabilities_into_slice(np, ni, matrix, log_probabilities, p.as_mut());
        }
        6 => {
            let mut p = std::ptr::NonNull::new(prior_ptr as *mut TRPParameters).unwrap();
            log_probabilities_into_slice(np, ni, matrix, log_probabilities, p.as_mut());
        }
        7 => {
            let mut p = std::ptr::NonNull::new(prior_ptr as *mut URPParameters).unwrap();
            log_probabilities_into_slice(np, ni, matrix, log_probabilities, p.as_mut());
        }
        _ => panic!("Unsupported prior ID: {}", prior_id),
    };
}
