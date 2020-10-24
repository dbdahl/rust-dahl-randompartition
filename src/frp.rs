// Focal random partition distribution

use crate::clust::{Clustering, Permutation};
use crate::mcmc::PriorLogWeight;
use crate::prelude::*;

use dahl_roxido::mk_rng_isaac;
use rand::prelude::*;
use rand_isaac::IsaacRng;
use std::convert::TryFrom;
use std::slice;

pub struct FRPParameters {
    focal: Clustering,
    weights: Weights,
    permutation: Permutation,
    mass: Mass,
    discount: Discount,
}

impl FRPParameters {
    pub fn new(
        focal: Clustering,
        weights: Weights,
        permutation: Permutation,
        mass: Mass,
        discount: Discount,
    ) -> Option<Self> {
        if weights.len() != focal.n_items() {
            None
        } else if focal.n_items() != permutation.len() {
            None
        } else {
            Some(Self {
                focal: focal.standardize(),
                weights,
                permutation,
                mass,
                discount,
            })
        }
    }

    pub fn shuffle_permutation<T: Rng>(&mut self, rng: &mut T) {
        self.permutation.shuffle(rng);
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

impl PriorLogWeight for FRPParameters {
    fn log_weight(&self, item_index: usize, subset_index: usize, clustering: &Clustering) -> f64 {
        let mut p = clustering.clone();
        p.reallocate(item_index, subset_index);
        log_pmf(&p, self)
    }
}

pub fn engine<T: Rng>(
    parameters: &FRPParameters,
    target: Option<&Clustering>,
    mut rng: Option<&mut T>,
) -> (Clustering, f64) {
    let ni = parameters.focal.n_items();
    let mass = parameters.mass.unwrap();
    let discount = parameters.discount.unwrap();
    let p: Clustering;
    let target = match target {
        Some(target) => {
            p = target.standardize_by(&parameters.permutation);
            Some(&p)
        }
        None => None,
    };
    let mut log_probability = 0.0;
    let mut clustering = Clustering::unallocated(ni);
    let mut total_counter = vec![0.0; parameters.focal.new_label()];
    let mut intersection_counter = Vec::with_capacity(total_counter.len());
    for _ in 0..total_counter.len() {
        intersection_counter.push(Vec::new())
    }
    for i in 0..clustering.n_items() {
        let ii = parameters.permutation.get(i);
        let focal_subset_index = parameters.focal[ii];
        let scaled_weight = (i as f64) * parameters.weights[ii];
        let normalized_scaled_weight = if total_counter[focal_subset_index] == 0.0 {
            0.0
        } else {
            scaled_weight / total_counter[focal_subset_index]
        };
        let n_occupied_subsets = clustering.n_clusters() as f64;
        let labels_and_weights = clustering.available_labels_for_allocation().map(|label| {
            let n_items_in_cluster = clustering.size_of(label);
            let weight = if n_items_in_cluster == 0 {
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
                (n_items_in_cluster as f64) - discount
                    + normalized_scaled_weight * intersection_counter[focal_subset_index][label]
            };
            (label, weight)
        });
        //        println!( "i {}, ii {}\ntarget {:?}\nfocal {:?}\ncurrent {:?}", i, ii, target.unwrap(), parameters.focal, clustering );
        //        println!("{:?}", labels_and_weights.clone().collect::<Vec<_>>());
        let (subset_index, log_probability_contribution) = match &mut rng {
            Some(r) => clustering.select(labels_and_weights, false, 0, Some(r), true),
            None => clustering.select::<IsaacRng, _>(
                labels_and_weights,
                false,
                target.unwrap()[i],
                None,
                true,
            ),
        };
        //        println!("*");
        log_probability += log_probability_contribution;
        if subset_index == intersection_counter[0].len() {
            for counter in intersection_counter.iter_mut() {
                counter.push(0.0);
            }
        }
        intersection_counter[focal_subset_index][subset_index] += 1.0;
        total_counter[focal_subset_index] += 1.0;
        clustering.allocate(ii, subset_index);
    }
    if !parameters.permutation.natural {
        clustering = clustering.relabel(0, None, false).0;
    }
    (clustering, log_probability)
}

/*
pub fn engine2<T: Rng>(
    parameters: &FRPParameters,
    mut target_or_rng: TargetOrRandom2<T>,
) -> (Partition, f64) {
    let nsf = parameters.focal.n_subsets();
    let ni = parameters.focal.n_items();
    let mass = parameters.mass.unwrap();
    let discount = parameters.discount.unwrap();
    if let TargetOrRandom::Target(t) = &mut target_or_rng {
        assert_eq!(t.n_items(), ni);
        *t = t.standardize(0, Some(parameters.permutation)).0;
    };
    let mut log_probability = 0.0;
    let mut partition = Clustering::unallocated(ni);
    let mut total_counter = vec![0.0; nsf];
    let mut intersection_counter = Vec::with_capacity(nsf);
    for _ in 0..nsf {
        intersection_counter.push(Vec::new())
    }
    for i in 0..ni {
        let ii = parameters.permutation.get(i);
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
            TargetOrRandom::Target(t) => t.get(ii),
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
*/

pub fn sample<T: Rng>(parameters: &FRPParameters, rng: &mut T) -> Clustering {
    engine(parameters, None, Some(rng)).0
}

pub fn log_pmf(target: &Clustering, parameters: &FRPParameters) -> f64 {
    engine::<IsaacRng>(parameters, Some(target), None).1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goodness_of_fit_constructive() {
        let n_items = 4;
        let discount = 0.1;
        let mass = Mass::new_with_variable_constraint(2.0, discount);
        let discount = Discount::new(discount);
        let mut rng = thread_rng();
        for focal in Clustering::iter(n_items) {
            let focal = Clustering::from_vector(focal);
            let mut vec = Vec::with_capacity(focal.n_clusters());
            for _ in 0..focal.n_items() {
                vec.push(rng.gen_range(0.0, 10.0));
            }
            let weights = Weights::from(&vec[..]).unwrap();
            let permutation = Permutation::random(n_items, &mut rng);
            let parameters =
                FRPParameters::new(focal, weights, permutation, mass, discount).unwrap();
            let sample_closure = || sample(&parameters, &mut thread_rng());
            let log_prob_closure = |partition: &mut Clustering| log_pmf(partition, &parameters);
            crate::testing::assert_goodness_of_fit(
                10000,
                n_items,
                sample_closure,
                log_prob_closure,
                1,
                0.001,
            )
        }
    }

    #[test]
    fn test_pmf() {
        let n_items = 5;
        let discount = 0.1;
        let mass = Mass::new_with_variable_constraint(2.0, discount);
        let discount = Discount::new(discount);
        let mut rng = thread_rng();
        for focal in Clustering::iter(n_items) {
            let focal = Clustering::from_vector(focal);
            let mut vec = Vec::with_capacity(focal.n_clusters());
            for _ in 0..focal.n_items() {
                vec.push(rng.gen_range(0.0, 10.0));
            }
            let weights = Weights::from(&vec[..]).unwrap();
            let permutation = Permutation::random(n_items, &mut rng);
            let parameters =
                FRPParameters::new(focal, weights, permutation, mass, discount).unwrap();
            let log_prob_closure = |partition: &mut Clustering| log_pmf(partition, &parameters);
            crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__frpparameters_new(
    n_items: i32,
    focal_ptr: *const i32,
    weights_ptr: *const f64,
    permutation_ptr: *const i32,
    use_random_permutations: i32,
    mass: f64,
    discount: f64,
) -> *mut FRPParameters {
    let ni = n_items as usize;
    let focal = Clustering::from_slice(slice::from_raw_parts(focal_ptr, ni));
    let weights = Weights::from(slice::from_raw_parts(weights_ptr, ni)).unwrap();
    let permutation = if use_random_permutations != 0 {
        Permutation::natural(ni)
    } else {
        let permutation_slice = slice::from_raw_parts(permutation_ptr, ni);
        let permutation_vector: Vec<usize> =
            permutation_slice.iter().map(|x| *x as usize).collect();
        Permutation::from_vector(permutation_vector).unwrap()
    };
    let d = Discount::new(discount);
    let m = Mass::new_with_variable_constraint(mass, discount);
    // First we create a new object.
    let obj = FRPParameters::new(focal, weights, permutation, m, d).unwrap();
    // Then copy it to the heap (so we have a stable pointer to it).
    let boxed_obj = Box::new(obj);
    // Then return a pointer by converting our `Box<_>` into a raw pointer
    Box::into_raw(boxed_obj)
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__frpparameters_free(obj: *mut FRPParameters) {
    // As a rule of thumb, freeing a null pointer is just a noop.
    if obj.is_null() {
        return;
    }
    // Convert the raw pointer back to a Box<_>
    let boxed = Box::from_raw(obj);
    // Then explicitly drop it (optional)
    drop(boxed);
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
    let focal = Clustering::from_slice(slice::from_raw_parts(focal_ptr, ni));
    let weights = Weights::from(slice::from_raw_parts(weights_ptr, ni)).unwrap();
    let permutation = if use_random_permutations != 0 {
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
    let mut parameters = FRPParameters::new(focal, weights, permutation, mass, discount).unwrap();
    if do_sampling != 0 {
        let mut rng = &mut mk_rng_isaac(seed_ptr);
        for i in 0..np {
            if use_random_permutations != 0 {
                parameters.shuffle_permutation(&mut rng);
            }
            let p = engine(&parameters, None, Some(rng));
            let labels = p.0.labels();
            for j in 0..ni {
                matrix[np * j + i] = i32::try_from(labels[j] + 1).unwrap();
            }
            probs[i] = p.1;
        }
    } else {
        for i in 0..np {
            let mut target_labels = Vec::with_capacity(ni);
            for j in 0..ni {
                target_labels.push(matrix[np * j + i] as usize);
            }
            let target = Clustering::from_vector(target_labels);
            let p = engine::<IsaacRng>(&parameters, Some(&target), None);
            probs[i] = p.1;
        }
    }
}
