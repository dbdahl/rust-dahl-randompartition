use crate::clust::Clustering;
use crate::crp::CRPParameters;
use crate::frp::FRPParameters;
use crate::lsp::LSPParameters;
use dahl_roxido::mk_rng_isaac;
use rand::prelude::*;
use rand_isaac::IsaacRng;
use std::convert::TryFrom;
use std::ffi::c_void;
use std::slice;

pub trait PriorSampler {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering;
}

pub fn sample_into_slice<S: PriorSampler, T: Rng, F: Fn(&mut S, &mut T) -> ()>(
    n_partitions: usize,
    n_items: usize,
    matrix: &mut [i32],
    rng: &mut T,
    prior: &mut S,
    callback: F,
) {
    for i in 0..n_partitions {
        callback(prior, rng);
        let p = prior.sample(rng).standardize();
        let labels = p.allocation();
        for j in 0..n_items {
            matrix[n_partitions * j + i] = i32::try_from(labels[j] + 1).unwrap();
        }
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
    use_random_permutation: bool,
) -> () {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let matrix: &mut [i32] = slice::from_raw_parts_mut(partition_labels_ptr, np * ni);
    let rng = &mut mk_rng_isaac(seed_ptr);
    match prior_id {
        0 => {
            let mut p = std::ptr::NonNull::new(prior_ptr as *mut CRPParameters).unwrap();
            let callback = |_p: &mut CRPParameters, _rng: &mut IsaacRng| {};
            sample_into_slice(np, ni, matrix, rng, p.as_mut(), callback);
        }
        1 => {
            let mut p = std::ptr::NonNull::new(prior_ptr as *mut FRPParameters).unwrap();
            if use_random_permutation {
                let callback = |p: &mut FRPParameters, rng: &mut IsaacRng| {
                    p.shuffle_permutation(rng);
                };
                sample_into_slice(np, ni, matrix, rng, p.as_mut(), callback);
            } else {
                let callback = |_p: &mut FRPParameters, _rng: &mut IsaacRng| {};
                sample_into_slice(np, ni, matrix, rng, p.as_mut(), callback);
            }
        }
        2 => {
            let mut p = std::ptr::NonNull::new(prior_ptr as *mut LSPParameters).unwrap();
            if use_random_permutation {
                let callback = |p: &mut LSPParameters, rng: &mut IsaacRng| {
                    p.shuffle_permutation(rng);
                };
                sample_into_slice(np, ni, matrix, rng, p.as_mut(), callback);
            } else {
                let callback = |_p: &mut LSPParameters, _rng: &mut IsaacRng| {};
                sample_into_slice(np, ni, matrix, rng, p.as_mut(), callback);
            }
        }
        3 => {
            panic!("Cannot sample from the CPP ({})", prior_id);
        }
        _ => panic!("Unsupported prior ID: {}", prior_id),
    };
}
