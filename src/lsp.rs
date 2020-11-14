// Location scale partition distribution

use crate::clust::{Clustering, Permutation};
use crate::mcmc::PriorLogWeight;
use crate::prelude::*;
use crate::prior::PriorSampler;

use dahl_roxido::mk_rng_isaac;
use rand::prelude::*;
use rand_isaac::IsaacRng;
use std::convert::TryFrom;
use std::slice;

pub struct LSPParameters {
    location: Clustering,
    scale: Scale,
    rate: Rate,
    permutation: Permutation,
}

impl LSPParameters {
    pub fn new_with_scale(
        location: Clustering,
        scale: Scale,
        permutation: Permutation,
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
        location: Clustering,
        rate: Rate,
        permutation: Permutation,
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

    pub fn shuffle_permutation<T: Rng>(&mut self, rng: &mut T) {
        self.permutation.shuffle(rng);
    }
}

impl PriorLogWeight for LSPParameters {
    fn log_weight(&self, item_index: usize, subset_index: usize, clustering: &Clustering) -> f64 {
        let mut p = clustering.allocation().clone();
        p[item_index] = subset_index;
        engine::<IsaacRng>(self, Some(&p[..]), None).1
    }
}

impl PriorSampler for LSPParameters {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering {
        engine(self, None, Some(rng)).0
    }
}

fn engine<T: Rng>(
    parameters: &LSPParameters,
    target: Option<&[usize]>,
    mut rng: Option<&mut T>,
) -> (Clustering, f64) {
    let ni = parameters.location.n_items();
    let mut log_probability = 0.0;
    let mut clustering = Clustering::unallocated(ni);
    let mut total_counter = vec![0.0; parameters.location.max_label() + 1];
    let mut intersection_counter = Vec::with_capacity(total_counter.len());
    for _ in 0..total_counter.len() {
        intersection_counter.push(Vec::new())
    }
    let mut n_visited_subsets = 0;
    let mut visited_subsets_indicator = vec![false; total_counter.len()];
    for i in 0..clustering.n_items() {
        let ii = parameters.permutation.get(i);
        let location_subset_index = parameters.location[ii];
        let n_occupied_subsets = clustering.n_clusters() as f64;
        let labels_and_weights = clustering
            .available_labels_for_allocation_with_target(target, ii)
            .map(|label| {
                let n_items_in_cluster = clustering.size_of(label);
                let weight = if n_items_in_cluster == 0 {
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
                    (1.0 + parameters.rate * intersection_counter[location_subset_index][label])
                        / (1.0
                            + (n_visited_subsets as f64)
                            + parameters.rate * (n_items_in_cluster as f64))
                };
                (label, weight)
            });
        let (subset_index, log_probability_contribution) = match &mut rng {
            Some(r) => clustering.select(labels_and_weights, false, 0, Some(r), true),
            None => clustering.select::<IsaacRng, _>(
                labels_and_weights,
                false,
                target.unwrap()[ii],
                None,
                true,
            ),
        };
        log_probability += log_probability_contribution;
        if subset_index >= intersection_counter[0].len() {
            for counter in intersection_counter.iter_mut() {
                counter.resize(subset_index + 1, 0.0);
            }
        }
        intersection_counter[location_subset_index][subset_index] += 1.0;
        total_counter[location_subset_index] += 1.0;
        clustering.allocate(ii, subset_index);
        if !visited_subsets_indicator[location_subset_index] {
            n_visited_subsets += 1;
            visited_subsets_indicator[location_subset_index] = true;
        }
    }
    (clustering, log_probability)
}

pub fn sample<T: Rng>(parameters: &LSPParameters, rng: &mut T) -> Clustering {
    engine(parameters, None, Some(rng)).0
}

pub fn log_pmf(target: &Clustering, parameters: &LSPParameters) -> f64 {
    engine::<IsaacRng>(parameters, Some(target.allocation()), None).1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goodness_of_fit_constructive() {
        let n_items = 4;
        let mut rng = thread_rng();
        for location in Clustering::iter(n_items) {
            let location = Clustering::from_vector(location);
            let rate = Rate::new(rng.gen_range(0.0, 10.0));
            let permutation = Permutation::random(n_items, &mut rng);
            let parameters = LSPParameters::new_with_rate(location, rate, permutation).unwrap();
            let sample_closure = || sample(&parameters, &mut thread_rng());
            let log_prob_closure = |clustering: &mut Clustering| log_pmf(clustering, &parameters);
            crate::testing::assert_goodness_of_fit(
                10000,
                n_items,
                sample_closure,
                log_prob_closure,
                1,
                0.001,
            );
        }
    }

    #[test]
    fn test_pmf() {
        let n_items = 5;
        let mut rng = thread_rng();
        for location in Clustering::iter(n_items) {
            let location = Clustering::from_vector(location);
            let rate = Rate::new(rng.gen_range(0.0, 10.0));
            let permutation = Permutation::random(n_items, &mut rng);
            let parameters = LSPParameters::new_with_rate(location, rate, permutation).unwrap();
            let log_prob_closure = |clustering: &mut Clustering| log_pmf(clustering, &parameters);
            crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__lspparameters_new(
    n_items: i32,
    focal_ptr: *const i32,
    rate: f64,
    permutation_ptr: *const i32,
    use_random_permutations: i32,
) -> *mut LSPParameters {
    let ni = n_items as usize;
    let focal = Clustering::from_slice(slice::from_raw_parts(focal_ptr, ni));
    let permutation = if use_random_permutations != 0 {
        Permutation::natural(ni)
    } else {
        let permutation_slice = slice::from_raw_parts(permutation_ptr, ni);
        let permutation_vector: Vec<usize> =
            permutation_slice.iter().map(|x| *x as usize).collect();
        Permutation::from_vector(permutation_vector).unwrap()
    };
    let r = Rate::new(rate);
    // First we create a new object.
    let obj = LSPParameters::new_with_rate(focal, r, permutation).unwrap();
    // Then copy it to the heap (so we have a stable pointer to it).
    let boxed_obj = Box::new(obj);
    // Then return a pointer by converting our `Box<_>` into a raw pointer
    Box::into_raw(boxed_obj)
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__lspparameters_free(obj: *mut LSPParameters) {
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
    let location = Clustering::from_slice(slice::from_raw_parts(location_ptr, ni));
    let rate = Rate::new(rate);
    let permutation = if use_random_permutations != 0 {
        Permutation::natural(ni)
    } else {
        let permutation_slice = slice::from_raw_parts(permutation_ptr, ni);
        let permutation_vector: Vec<usize> =
            permutation_slice.iter().map(|x| *x as usize).collect();
        Permutation::from_vector(permutation_vector).unwrap()
    };
    let matrix: &mut [i32] = slice::from_raw_parts_mut(partition_labels_ptr, np * ni);
    let probs: &mut [f64] = slice::from_raw_parts_mut(partition_probs_ptr, np);
    let mut parameters = LSPParameters::new_with_rate(location, rate, permutation).unwrap();
    if do_sampling != 0 {
        let mut rng = &mut mk_rng_isaac(seed_ptr);
        for i in 0..np {
            if use_random_permutations != 0 {
                parameters.shuffle_permutation(&mut rng);
            }
            let p = engine(&parameters, None, Some(rng));
            let labels = p.0.allocation();
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
            let p = engine::<IsaacRng>(&parameters, Some(target.allocation()), None);
            probs[i] = p.1;
        }
    }
}
