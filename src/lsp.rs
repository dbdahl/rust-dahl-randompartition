// Location scale partition distribution

use crate::mcmc::PriorLogWeight;
use crate::prelude::*;
use crate::TargetOrRandom;

use dahl_partition::*;
use dahl_roxido::mk_rng_isaac;
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::*;
use rand_isaac::IsaacRng;
use std::convert::TryFrom;
use std::slice;

pub struct LSPParameters<'a, 'c> {
    location: &'a Partition,
    scale: Scale,
    rate: Rate,
    permutation: &'c Permutation,
}

impl<'a, 'c> LSPParameters<'a, 'c> {
    pub fn new_with_scale(
        location: &'a Partition,
        scale: Scale,
        permutation: &'c Permutation,
    ) -> Option<Self> {
        if location.n_items() != permutation.len() {
            None
        } else {
            Some(Self {
                location,
                scale,
                rate: Rate::new(1.0 / scale.unwrap()),
                permutation,
            })
        }
    }
    pub fn new_with_rate(
        location: &'a Partition,
        rate: Rate,
        permutation: &'c Permutation,
    ) -> Option<Self> {
        if location.n_items() != permutation.len() {
            None
        } else {
            Some(Self {
                location,
                scale: Scale::new(1.0 / rate.unwrap()),
                rate,
                permutation,
            })
        }
    }
}

impl<'a, 'c> PriorLogWeight for LSPParameters<'a, 'c> {
    fn log_weight(&self, item_index: usize, subset_index: usize, partition: &Partition) -> f64 {
        let mut p = partition.clone();
        p.add_with_index(item_index, subset_index);
        log_pmf_mut(&mut p, self)
    }
}

pub fn engine<T: Rng>(
    parameters: &LSPParameters,
    mut target_or_rng: TargetOrRandom<T>,
) -> (Partition, f64) {
    let nsf = parameters.location.n_subsets();
    let ni = parameters.location.n_items();
    if let TargetOrRandom::Target(t) = &mut target_or_rng {
        assert_eq!(t.n_items(), ni);
        t.canonicalize_by_permutation(Some(&parameters.permutation));
    };
    let mut log_probability = 0.0;
    let mut partition = Partition::new(ni);
    let mut total_counter = vec![0.0; nsf];
    let mut intersection_counter = Vec::with_capacity(nsf);
    for _ in 0..nsf {
        intersection_counter.push(Vec::new())
    }
    let mut n_visited_subsets = 0;
    let mut visited_subsets_indicator = vec![false; nsf];
    for i in 0..ni {
        let ii = parameters.permutation[i];
        let location_subset_index = parameters.location.label_of(ii).unwrap();
        // Ensure there is an empty subset
        match partition.subsets().last() {
            None => partition.new_subset(),
            Some(last) => {
                if !last.is_empty() {
                    partition.new_subset()
                }
            }
        }
        let n_occupied_subsets = (partition.n_subsets() - 1) as f64;
        let probs: Vec<(usize, f64)> = partition
            .subsets()
            .iter()
            .enumerate()
            .map(|(subset_index, subset)| {
                let prob = if subset.is_empty() {
                    if n_occupied_subsets == 0.0 {
                        1.0
                    } else {
                        {
                            if total_counter[location_subset_index] == 0.0 {
                                (1.0 + parameters.rate)
                                    / (1.0 + (n_visited_subsets as f64) + parameters.rate)
                            } else {
                                1.0 / (1.0 + (n_visited_subsets as f64) + parameters.rate)
                            }
                        }
                    }
                } else {
                    (1.0 + parameters.rate
                        * intersection_counter[location_subset_index][subset_index])
                        / (1.0
                            + (n_visited_subsets as f64)
                            + parameters.rate * (subset.n_items() as f64))
                };
                (subset_index, prob)
            })
            .collect();
        let subset_index = match &mut target_or_rng {
            TargetOrRandom::Random(rng) => {
                let dist = WeightedIndex::new(probs.iter().map(|x| x.1)).unwrap();
                dist.sample(*rng)
            }
            TargetOrRandom::Target(t) => t.label_of(ii).unwrap(),
        };
        let numerator = probs[subset_index].1;
        let denominator = probs.iter().fold(0.0, |sum, x| sum + x.1);
        log_probability += (numerator / denominator).ln();
        if subset_index == intersection_counter[0].len() {
            for counter in intersection_counter.iter_mut() {
                counter.push(0.0);
            }
        }
        intersection_counter[location_subset_index][subset_index] += 1.0;
        total_counter[location_subset_index] += 1.0;
        partition.add_with_index(ii, subset_index);
        if !visited_subsets_indicator[location_subset_index] {
            n_visited_subsets += 1;
            visited_subsets_indicator[location_subset_index] = true;
        }
    }
    partition.canonicalize();
    (partition, log_probability)
}

pub fn sample<T: Rng>(parameters: &LSPParameters, rng: &mut T) -> Partition {
    engine(parameters, TargetOrRandom::Random(rng)).0
}

pub fn log_pmf(target: &Partition, parameters: &LSPParameters) -> f64 {
    let mut target2 = target.clone();
    engine(parameters, TargetOrRandom::Target::<IsaacRng>(&mut target2)).1
}

pub fn log_pmf_mut(target: &mut Partition, parameters: &LSPParameters) -> f64 {
    engine(parameters, TargetOrRandom::Target::<IsaacRng>(target)).1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goodness_of_fit_constructive() {
        let n_items = 4;
        let mut permutation = Permutation::natural(n_items);
        let mut rng = thread_rng();
        for location in Partition::iter(n_items) {
            permutation.shuffle(&mut rng);
            let location = Partition::from(&location[..]);
            let rate = Rate::new(rng.gen_range(0.0, 10.0));
            let parameters = LSPParameters::new_with_rate(&location, rate, &permutation).unwrap();
            let sample_closure = || sample(&parameters, &mut thread_rng());
            let log_prob_closure = |partition: &mut Partition| log_pmf(partition, &parameters);
            if let Some(string) = crate::testing::assert_goodness_of_fit(
                10000,
                n_items,
                sample_closure,
                log_prob_closure,
                1,
                0.001,
            ) {
                panic!("{}", string);
            }
        }
    }

    #[test]
    fn test_pmf() {
        let n_items = 5;
        let mut permutation = Permutation::natural(n_items);
        let mut rng = thread_rng();
        for location in Partition::iter(n_items) {
            permutation.shuffle(&mut rng);
            let location = Partition::from(&location[..]);
            let rate = Rate::new(rng.gen_range(0.0, 10.0));
            let parameters = LSPParameters::new_with_rate(&location, rate, &permutation).unwrap();
            let log_prob_closure = |partition: &mut Partition| log_pmf(partition, &parameters);
            crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__ls_partition(
    do_sampling: i32,
    n_partitions: i32,
    n_items: i32,
    partition_labels_ptr: *mut i32,
    partition_probs_ptr: *mut f64,
    seed_ptr: *const i32, // Assumed length is 32
    location_ptr: *const i32,
    rate: f64,
    permutation_ptr: *const i32,
    use_random_permutations: i32,
) -> () {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let location = Partition::from(slice::from_raw_parts(location_ptr, ni));
    let rate = Rate::new(rate);
    let mut permutation = if use_random_permutations != 0 {
        Permutation::natural(ni)
    } else {
        let permutation_slice = slice::from_raw_parts(permutation_ptr, ni);
        let permutation_vector: Vec<usize> =
            permutation_slice.iter().map(|x| *x as usize).collect();
        Permutation::from_vector(permutation_vector).unwrap()
    };
    let matrix: &mut [i32] = slice::from_raw_parts_mut(partition_labels_ptr, np * ni);
    let probs: &mut [f64] = slice::from_raw_parts_mut(partition_probs_ptr, np);
    if do_sampling != 0 {
        let mut rng = &mut mk_rng_isaac(seed_ptr);
        for i in 0..np {
            if use_random_permutations != 0 {
                permutation.shuffle(&mut rng);
            }
            let parameters = LSPParameters::new_with_rate(&location, rate, &permutation).unwrap();
            let p = engine(&parameters, TargetOrRandom::Random(rng));
            let labels = p.0.labels();
            for j in 0..ni {
                matrix[np * j + i] = i32::try_from(labels[j].unwrap() + 1).unwrap();
            }
            probs[i] = p.1;
        }
    } else {
        let parameters = LSPParameters::new_with_rate(&location, rate, &permutation).unwrap();
        for i in 0..np {
            let mut target_labels = Vec::with_capacity(ni);
            for j in 0..ni {
                target_labels.push(matrix[np * j + i]);
            }
            let mut target = Partition::from(&target_labels[..]);
            let p = engine::<IsaacRng>(&parameters, TargetOrRandom::Target(&mut target));
            probs[i] = p.1;
        }
    }
}
