// Location scale partition distribution

use crate::clust::Clustering;
use crate::distr::{PartitionSampler, PredictiveProbabilityFunctionOld};
use crate::perm::Permutation;
use crate::prelude::*;
use crate::prior::PartitionLogProbability;

use rand::prelude::*;
use rand_isaac::IsaacRng;
use std::slice;

#[derive(Debug, Clone)]
pub struct LspParameters {
    pub baseline: Clustering,
    pub scale: Scale,
    pub rate: Rate,
    pub permutation: Permutation,
}

impl LspParameters {
    pub fn new_with_scale(
        baseline: Clustering,
        scale: Scale,
        permutation: Permutation,
    ) -> Option<Self> {
        if baseline.n_items() != permutation.n_items() {
            None
        } else {
            Some(Self {
                baseline,
                scale,
                rate: Rate::new(1.0 / scale.unwrap()),
                permutation,
            })
        }
    }
    pub fn new_with_rate(
        baseline: Clustering,
        rate: Rate,
        permutation: Permutation,
    ) -> Option<Self> {
        if baseline.n_items() != permutation.n_items() {
            None
        } else {
            Some(Self {
                baseline,
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

impl PredictiveProbabilityFunctionOld for LspParameters {
    fn log_predictive_probability(
        &self,
        item_index: usize,
        subset_index: usize,
        clustering: &Clustering,
    ) -> f64 {
        let mut p = clustering.allocation().clone();
        p[item_index] = subset_index;
        engine::<IsaacRng>(self, Some(&p[..]), None).1
    }
}

impl PartitionSampler for LspParameters {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering {
        engine(self, None, Some(rng)).0
    }
}

impl PartitionLogProbability for LspParameters {
    fn log_probability(&self, partition: &Clustering) -> f64 {
        engine::<IsaacRng>(self, Some(partition.allocation()), None).1
    }
    fn is_normalized(&self) -> bool {
        true
    }
}

fn engine<T: Rng>(
    parameters: &LspParameters,
    target: Option<&[usize]>,
    mut rng: Option<&mut T>,
) -> (Clustering, f64) {
    let ni = parameters.baseline.n_items();
    let mut log_probability = 0.0;
    let mut clustering = Clustering::unallocated(ni);
    let mut total_counter = vec![0.0; parameters.baseline.max_label() + 1];
    let mut intersection_counter = Vec::with_capacity(total_counter.len());
    for _ in 0..total_counter.len() {
        intersection_counter.push(Vec::new())
    }
    let mut n_visited_subsets = 0;
    let mut visited_subsets_indicator = vec![false; total_counter.len()];
    for i in 0..clustering.n_items() {
        let ii = parameters.permutation.get(i);
        let baseline_subset_index = parameters.baseline[ii];
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
                            if total_counter[baseline_subset_index] == 0.0 {
                                (1.0 + parameters.rate)
                                    / (1.0 + (n_visited_subsets as f64) + parameters.rate)
                            } else {
                                1.0 / (1.0 + (n_visited_subsets as f64) + parameters.rate)
                            }
                        }
                    }
                } else {
                    (1.0 + parameters.rate * intersection_counter[baseline_subset_index][label])
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
        intersection_counter[baseline_subset_index][subset_index] += 1.0;
        total_counter[baseline_subset_index] += 1.0;
        clustering.allocate(ii, subset_index);
        if !visited_subsets_indicator[baseline_subset_index] {
            n_visited_subsets += 1;
            visited_subsets_indicator[baseline_subset_index] = true;
        }
    }
    (clustering, log_probability)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goodness_of_fit_constructive() {
        let n_items = 4;
        let mut rng = thread_rng();
        for baseline in Clustering::iter(n_items) {
            let baseline = Clustering::from_vector(baseline);
            let rate = Rate::new(rng.gen_range(0.0..10.0));
            let permutation = Permutation::random(n_items, &mut rng);
            let parameters = LspParameters::new_with_rate(baseline, rate, permutation).unwrap();
            let sample_closure = || parameters.sample(&mut thread_rng());
            let log_prob_closure =
                |clustering: &mut Clustering| parameters.log_probability(clustering);
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
        for baseline in Clustering::iter(n_items) {
            let baseline = Clustering::from_vector(baseline);
            let rate = Rate::new(rng.gen_range(0.0..10.0));
            let permutation = Permutation::random(n_items, &mut rng);
            let parameters = LspParameters::new_with_rate(baseline, rate, permutation).unwrap();
            let log_prob_closure =
                |clustering: &mut Clustering| parameters.log_probability(clustering);
            crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__lspparameters_new(
    n_items: i32,
    baseline_ptr: *const i32,
    rate: f64,
    permutation_ptr: *const i32,
    use_natural_permutation: i32,
) -> *mut LspParameters {
    let ni = n_items as usize;
    let baseline = Clustering::from_slice(slice::from_raw_parts(baseline_ptr, ni));
    let permutation = if use_natural_permutation != 0 {
        Permutation::natural_and_fixed(ni)
    } else {
        let permutation_slice = slice::from_raw_parts(permutation_ptr, ni);
        let permutation_vector: Vec<usize> =
            permutation_slice.iter().map(|x| *x as usize).collect();
        Permutation::from_vector(permutation_vector).unwrap()
    };
    let r = Rate::new(rate);
    // First we create a new object.
    let obj = LspParameters::new_with_rate(baseline, r, permutation).unwrap();
    // Then copy it to the heap (so we have a stable pointer to it).
    let boxed_obj = Box::new(obj);
    // Then return a pointer by converting our `Box<_>` into a raw pointer
    Box::into_raw(boxed_obj)
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__lspparameters_free(obj: *mut LspParameters) {
    // As a rule of thumb, freeing a null pointer is just a noop.
    if obj.is_null() {
        return;
    }
    // Convert the raw pointer back to a Box<_>
    let boxed = Box::from_raw(obj);
    // Then explicitly drop it (optional)
    drop(boxed);
}
