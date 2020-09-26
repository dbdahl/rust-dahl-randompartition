// Focal random partition distribution

use crate::mcmc::NealFunctionsGeneral;
use crate::prelude::*;
use crate::TargetOrRandom;

use dahl_partition::*;
use dahl_roxido::mk_rng_isaac;
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::*;
use rand_isaac::IsaacRng;
use std::convert::TryFrom;
use std::slice;

pub struct FRPParameters<'a, 'b, 'c> {
    focal: &'a Partition,
    weights: &'b Weights,
    permutation: &'c Permutation,
    mass: Mass,
    discount: Discount,
}

impl<'a, 'b, 'c> FRPParameters<'a, 'b, 'c> {
    pub fn new(
        focal: &'a Partition,
        weights: &'b Weights,
        permutation: &'c Permutation,
        mass: Mass,
        discount: Discount,
    ) -> Option<Self> {
        if weights.len() != focal.n_items() {
            None
        } else if focal.n_items() != permutation.len() {
            None
        } else {
            Some(Self {
                focal,
                weights,
                permutation,
                mass,
                discount,
            })
        }
    }
}

#[derive(Debug)]
pub struct Weights(Vec<f64>);

impl Weights {
    pub fn zero(n_items: usize) -> Weights {
        Weights(vec![0.0; n_items])
    }

    pub fn from_rate(rate: Rate, n_items: usize) -> Weights {
        Weights(vec![rate.unwrap(); n_items])
    }

    pub fn constant(value: f64, n_items: usize) -> Option<Weights> {
        if value.is_nan() || value.is_infinite() || value < 0.0 {
            return None;
        }
        Some(Weights(vec![value; n_items]))
    }

    pub fn from(w: &[f64]) -> Option<Weights> {
        for ww in w.iter() {
            if ww.is_nan() || ww.is_infinite() || *ww < 0.0 {
                return None;
            }
        }
        Some(Weights(Vec::from(w)))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl std::ops::Index<usize> for Weights {
    type Output = f64;
    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}

impl<'a, 'b, 'c> NealFunctionsGeneral for FRPParameters<'a, 'b, 'c> {
    fn new_weight(&self, _n_subsets: usize) -> f64 {
        self.mass.unwrap()
    }

    fn existing_weight(&self, _item_index: usize, _subset: &Subset, _partition: &Partition) -> f64 {
        panic!("No implemented!");
    }

    fn weight(&self, item_index: usize, subset_index: usize, partition: &Partition) -> f64 {
        let mut p = partition.clone();
        p.add_with_index(item_index, subset_index);
        log_pmf(&p, self).exp()
    }
}

pub fn engine<T: Rng>(
    parameters: &FRPParameters,
    mut target_or_rng: TargetOrRandom<T>,
) -> (Partition, f64) {
    let nsf = parameters.focal.n_subsets();
    let ni = parameters.focal.n_items();
    let mass = parameters.mass.unwrap();
    let discount = parameters.discount.unwrap();
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
    for i in 0..ni {
        let ii = parameters.permutation[i];
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
        let focal_subset_index = parameters.focal.label_of(ii).unwrap();
        let scaled_weight = (i as f64) * parameters.weights[ii];

        // This code chunk should be deleted in production.
        let key = "DBD_PUMPKIN_SCALING";
        let scaled_weight = match std::env::var(key) {
            Ok(val) => {
                if val.to_lowercase() == "old" {
                    match std::env::var("DBD_PUMPKIN_VERBOSE") {
                        Ok(val) => {
                            if val.to_lowercase() == "true" {
                                println!("Using old weighting method.");
                            }
                        }
                        Err(_e) => {}
                    }
                    parameters.weights[ii]
                } else {
                    scaled_weight
                }
            }
            Err(_e) => scaled_weight,
        };

        let normalized_scaled_weight = if total_counter[focal_subset_index] == 0.0 {
            0.0
        } else {
            scaled_weight / total_counter[focal_subset_index]
        };
        let probs: Vec<(usize, f64)> = partition
            .subsets()
            .iter()
            .enumerate()
            .map(|(subset_index, subset)| {
                let prob = if subset.is_empty() {
                    if n_occupied_subsets == 0.0 {
                        1.0
                    } else {
                        mass + discount * n_occupied_subsets + {
                            if total_counter[focal_subset_index] == 0.0 {
                                scaled_weight
                            } else {
                                0.0
                            }
                        }
                    }
                } else {
                    (subset.n_items() as f64) - discount
                        + normalized_scaled_weight
                            * intersection_counter[focal_subset_index][subset_index]
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
        intersection_counter[focal_subset_index][subset_index] += 1.0;
        total_counter[focal_subset_index] += 1.0;
        partition.add_with_index(ii, subset_index);
    }
    partition.canonicalize();
    (partition, log_probability)
}

pub fn sample<T: Rng>(parameters: &FRPParameters, rng: &mut T) -> Partition {
    engine(parameters, TargetOrRandom::Random(rng)).0
}

pub fn log_pmf(target: &Partition, parameters: &FRPParameters) -> f64 {
    let mut target2 = target.clone();
    engine(parameters, TargetOrRandom::Target::<IsaacRng>(&mut target2)).1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goodness_of_fit_constructive() {
        let n_items = 4;
        let mut permutation = Permutation::natural(n_items);
        let discount = 0.1;
        let mass = Mass::new_with_variable_constraint(2.0, discount);
        let discount = Discount::new(discount);
        let mut rng = thread_rng();
        for focal in Partition::iter(n_items) {
            permutation.shuffle(&mut rng);
            let focal = Partition::from(&focal[..]);
            let mut vec = Vec::with_capacity(focal.n_subsets());
            for _ in 0..focal.n_items() {
                vec.push(rng.gen_range(0.0, 10.0));
            }
            let weights = Weights::from(&vec[..]).unwrap();
            let parameters =
                FRPParameters::new(&focal, &weights, &permutation, mass, discount).unwrap();
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
        let discount = 0.1;
        let mass = Mass::new_with_variable_constraint(2.0, discount);
        let discount = Discount::new(discount);
        let mut rng = thread_rng();
        for focal in Partition::iter(n_items) {
            permutation.shuffle(&mut rng);
            let focal = Partition::from(&focal[..]);
            let mut vec = Vec::with_capacity(focal.n_subsets());
            for _ in 0..focal.n_items() {
                vec.push(rng.gen_range(0.0, 10.0));
            }
            let weights = Weights::from(&vec[..]).unwrap();
            let parameters =
                FRPParameters::new(&focal, &weights, &permutation, mass, discount).unwrap();
            let log_prob_closure = |partition: &mut Partition| log_pmf(partition, &parameters);
            crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__focal_partition(
    do_sampling: i32,
    n_partitions: i32,
    n_items: i32,
    partition_labels_ptr: *mut i32,
    partition_probs_ptr: *mut f64,
    seed_ptr: *const i32, // Assumed length is 32
    focal_ptr: *const i32,
    weights_ptr: *const f64,
    permutation_ptr: *const i32,
    mass: f64,
    discount: f64,
    use_random_permutations: i32,
) -> () {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let focal = Partition::from(slice::from_raw_parts(focal_ptr, ni));
    let weights = Weights::from(slice::from_raw_parts(weights_ptr, ni)).unwrap();
    let mut permutation = if use_random_permutations != 0 {
        Permutation::natural(ni)
    } else {
        let permutation_slice = slice::from_raw_parts(permutation_ptr, ni);
        let permutation_vector: Vec<usize> =
            permutation_slice.iter().map(|x| *x as usize).collect();
        Permutation::from_vector(permutation_vector).unwrap()
    };
    let mass = Mass::new_with_variable_constraint(mass, discount);
    let discount = Discount::new(discount);
    let matrix: &mut [i32] = slice::from_raw_parts_mut(partition_labels_ptr, np * ni);
    let probs: &mut [f64] = slice::from_raw_parts_mut(partition_probs_ptr, np);
    if do_sampling != 0 {
        let mut rng = &mut mk_rng_isaac(seed_ptr);
        for i in 0..np {
            if use_random_permutations != 0 {
                permutation.shuffle(&mut rng);
            }
            let parameters =
                FRPParameters::new(&focal, &weights, &permutation, mass, discount).unwrap();
            let p = engine(&parameters, TargetOrRandom::Random(rng));
            let labels = p.0.labels();
            for j in 0..ni {
                matrix[np * j + i] = i32::try_from(labels[j].unwrap()).unwrap();
            }
            probs[i] = p.1;
        }
    } else {
        let parameters =
            FRPParameters::new(&focal, &weights, &permutation, mass, discount).unwrap();
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
