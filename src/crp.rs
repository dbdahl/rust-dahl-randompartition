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
    use crate::mcmc::{update_neal_algorithm3, update_rwmh};
    use rand::prelude::*;

    #[test]
    fn test_goodness_of_fit_constructive() {
        let n_items = 5;
        let parameters = CRPParameters::new(Mass::new(2.0));
        let sample_closure = || sample(n_items, &parameters, &mut thread_rng());
        let log_prob_closure = |partition: &mut Partition| log_pmf(partition, &parameters);
        crate::testing::assert_goodness_of_fit(
            100000,
            n_items,
            sample_closure,
            log_prob_closure,
            1,
            0.001,
        );
    }

    #[test]
    fn test_goodness_of_fit_neal_algorithm3() {
        let n_items = 5;
        let parameters = CRPParameters::new(Mass::new(2.0));
        let l = |_i: usize, _indices: &[usize]| 0.0;
        let mut p = Partition::one_subset(n_items);
        let sample_closure = || {
            p = update_neal_algorithm3(1, &p, &parameters, &l, &mut thread_rng());
            p.clone()
        };
        let log_prob_closure = |partition: &mut Partition| log_pmf(partition, &parameters);
        crate::testing::assert_goodness_of_fit(
            100000,
            n_items,
            sample_closure,
            log_prob_closure,
            5,
            0.001,
        );
    }

    #[test]
    fn test_goodness_of_fit_rwmh() {
        let n_items = 5;
        let parameters = CRPParameters::new(Mass::new(2.0));
        let log_prob_closure = |partition: &mut Partition| log_pmf(partition, &parameters);
        let log_prob_closure2 = |partition: &Partition| log_pmf(partition, &parameters);
        let rate = Rate::new(1.0);
        let mass = Mass::new(2.5); // Notice that the mass for the proposal doesn't need to match the prior
        let mut p = Partition::one_subset(n_items);
        let sample_closure = || {
            p = update_rwmh(1, &p, rate, mass, &log_prob_closure2, &mut thread_rng()).0;
            p.clone()
        };
        crate::testing::assert_goodness_of_fit(
            10000,
            n_items,
            sample_closure,
            log_prob_closure,
            5,
            0.001,
        );
    }

    #[test]
    fn test_pmf() {
        let parameters = CRPParameters::new(Mass::new(1.5));
        let log_prob_closure = |partition: &mut Partition| log_pmf(partition, &parameters);
        crate::testing::assert_pmf_sums_to_one(5, log_prob_closure, 0.0000001);
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
