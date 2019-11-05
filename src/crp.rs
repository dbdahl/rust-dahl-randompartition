// Chinese restaurant process

use crate::prelude::*;
use dahl_partition::*;
use rand::Rng;

use dahl_roxido::mk_rng_isaac;
use rand::distributions::{Distribution, WeightedIndex};
use statrs::function::gamma::ln_gamma;
use std::convert::TryFrom;
use std::slice;

pub fn sample<T: Rng>(n_items: usize, mass: Mass, rng: &mut T) -> Partition {
    let mut p = Partition::new(n_items);
    let mass = mass.as_f64();
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
    p
}

pub fn log_pmf(x: &Partition, mass: Mass) -> f64 {
    let ni = x.n_items() as f64;
    let ns = x.n_subsets() as f64;
    let m = mass.as_f64();
    let lm = mass.log();
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
        let mass = Mass::new(2.0);
        let mut samples = PartitionsHolder::with_capacity(n_partitions, n_items);
        for _ in 0..n_partitions {
            samples.push_partition(&sample(n_items, mass, &mut thread_rng()));
        }
        let mut psm = dahl_salso::psm::psm(&samples.view(), true);
        let truth = 1.0 / (1.0 + mass);
        let margin_of_error = 3.58 * (truth * (1.0 - truth) / n_partitions as f64).sqrt();
        assert!(psm.view().data().iter().all(|prob| {
            *prob == 1.0 || (truth - margin_of_error < *prob && *prob < truth + margin_of_error)
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__crp__sample(
    n_partitions: i32,
    n_items: i32,
    mass: f64,
    ptr: *mut i32,
    seed_ptr: *const i32, // Assumed length is 32
) -> () {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let mass = Mass::new(mass);
    let array: &mut [i32] = slice::from_raw_parts_mut(ptr, np * ni);
    let mut rng = mk_rng_isaac(seed_ptr);
    for i in 0..np {
        let p = sample(ni, mass, &mut rng);
        let labels = p.labels();
        for j in 0..ni {
            array[np * j + i] = i32::try_from(labels[j].unwrap()).unwrap();
        }
    }
}
