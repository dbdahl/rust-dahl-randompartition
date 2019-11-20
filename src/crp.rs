// Chinese restaurant process

use crate::mcmc::NealFunctions;
use crate::prelude::*;

use dahl_partition::*;
use dahl_roxido::mk_rng_isaac;
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use statrs::function::gamma::ln_gamma;
use std::convert::TryFrom;
use std::slice;

#[derive(Debug)]
pub struct CRPParameters {
    mass: Mass,
}

impl CRPParameters {
    pub fn new(mass: Mass) -> Self {
        Self { mass }
    }
}

impl NealFunctions for CRPParameters {
    fn new_weight(&self, _n_subsets: usize) -> f64 {
        self.mass.unwrap()
    }

    fn existing_weight(&self, _n_subsets: usize, n_items: usize) -> f64 {
        n_items as f64
    }
}

pub fn sample<T: Rng>(n_items: usize, parameters: &CRPParameters, rng: &mut T) -> Partition {
    let mut p = Partition::new(n_items);
    let mass = parameters.mass.unwrap();
    for i in 0..p.n_items() {
        match p.subsets().last() {
            None => p.new_subset(),
            Some(last) => {
                if !last.is_empty() {
                    p.new_subset()
                }
            }
        }
        let probs = p.subsets().iter().map(|subset| {
            if subset.is_empty() {
                mass
            } else {
                subset.n_items() as f64
            }
        });
        let dist = WeightedIndex::new(probs).unwrap();
        let subset_index = dist.sample(rng);
        p.add_with_index(i, subset_index);
    }
    p.canonicalize();
    p
}

pub fn log_pmf(x: &Partition, parameters: &CRPParameters) -> f64 {
    let ni = x.n_items() as f64;
    let ns = x.n_subsets() as f64;
    let m = parameters.mass.unwrap();
    let lm = m.ln();
    let mut result = ns * lm + ln_gamma(m) - ln_gamma(m + ni);
    for subset in x.subsets() {
        result += ln_gamma(subset.n_items() as f64);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_sample() {
        let n_partitions = 10000;
        let n_items = 4;
        let parameters = CRPParameters::new(Mass::new(2.0));
        let mut samples = PartitionsHolder::with_capacity(n_partitions, n_items);
        for _ in 0..n_partitions {
            samples.push_partition(&sample(n_items, &parameters, &mut thread_rng()));
        }
        let mut psm = dahl_salso::psm::psm(&samples.view(), true);
        let truth = 1.0 / (1.0 + parameters.mass);
        let margin_of_error = 3.58 * (truth * (1.0 - truth) / n_partitions as f64).sqrt();
        assert!(psm.view().data().iter().all(|prob| {
            *prob == 1.0 || (truth - margin_of_error < *prob && *prob < truth + margin_of_error)
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__crp_partition(
    do_sampling: i32,
    n_partitions: i32,
    n_items: i32,
    partition_labels_ptr: *mut i32,
    partition_probs_ptr: *mut f64,
    seed_ptr: *const i32, // Assumed length is 32
    mass: f64,
) -> () {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let parameters = CRPParameters::new(Mass::new(mass));
    let matrix: &mut [i32] = slice::from_raw_parts_mut(partition_labels_ptr, np * ni);
    let probs: &mut [f64] = slice::from_raw_parts_mut(partition_probs_ptr, np);
    if do_sampling != 0 {
        let rng = &mut mk_rng_isaac(seed_ptr);
        for i in 0..np {
            let p = sample(ni, &parameters, rng);
            let labels = p.labels();
            for j in 0..ni {
                matrix[np * j + i] = i32::try_from(labels[j].unwrap()).unwrap();
            }
            probs[i] = log_pmf(&p, &parameters);
        }
    } else {
        for i in 0..np {
            let mut target_labels = Vec::with_capacity(ni);
            for j in 0..ni {
                target_labels.push(matrix[np * j + i]);
            }
            let target = Partition::from(&target_labels[..]);
            probs[i] = log_pmf(&target, &parameters);
        }
    }
}
