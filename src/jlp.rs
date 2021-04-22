// Uniform partition distribution

use crate::clust::Clustering;
use crate::distr::{
    FullConditional, PartitionSampler, PredictiveProbabilityFunction, ProbabilityMassFunction,
};
use crate::perm::Permutation;
use crate::prelude::Mass;

use rand::Rng;
use std::slice;

#[derive(Debug, Clone)]
pub struct JlpParameters {
    n_items: usize,
    mass: Mass,
    permutation: Permutation,
}

impl JlpParameters {
    pub fn new(n_items: usize, mass: Mass, permutation: Permutation) -> Option<Self> {
        if n_items != permutation.n_items() {
            None
        } else {
            Some(Self {
                n_items,
                mass,
                permutation,
            })
        }
    }

    pub fn shuffle_permutation<T: Rng>(&mut self, rng: &mut T) {
        self.permutation.shuffle(rng);
    }
}

impl PredictiveProbabilityFunction for JlpParameters {
    fn log_predictive(
        &self,
        _item: usize,
        candidate_labels: &Vec<usize>,
        clustering: &Clustering,
    ) -> Vec<(usize, f64)> {
        candidate_labels
            .into_iter()
            .map(|label| {
                (
                    *label,
                    if clustering.size_of(*label) == 0 {
                        self.mass.ln()
                    } else {
                        0.0
                    },
                )
            })
            .collect()
    }
}

impl FullConditional for JlpParameters {
    fn log_full_conditional<'a>(
        &'a self,
        item: usize,
        clustering: &'a Clustering,
    ) -> Vec<(usize, f64)> {
        let mut working_clustering = clustering.clone();
        clustering
            .available_labels_for_reallocation(item)
            .map(|label| {
                (label, {
                    working_clustering.allocate(item, label);
                    self.log_pmf(&working_clustering)
                })
            })
            .collect()
    }
}

impl PartitionSampler for JlpParameters {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering {
        crate::distr::default_partition_sampler_sample(self, &self.permutation, rng)
    }
}

impl ProbabilityMassFunction for JlpParameters {
    fn log_pmf(&self, clustering: &Clustering) -> f64 {
        crate::distr::default_probability_mass_function_log_pmf(self, &self.permutation, clustering)
    }
    fn is_normalized(&self) -> bool {
        true
    }
}

mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_pmf() {
        let parameters =
            JlpParameters::new(5, Mass::new(2.0), Permutation::natural_and_fixed(5)).unwrap();
        let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
        crate::testing::assert_pmf_sums_to_one(parameters.n_items, log_prob_closure, 0.0000001);
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__jlpparameters_new(
    n_items: i32,
    mass: f64,
    permutation_ptr: *const i32,
    use_natural_permutation: i32,
) -> *mut JlpParameters {
    // First we create a new object.
    let ni = n_items as usize;
    let permutation = if use_natural_permutation != 0 {
        Permutation::natural_and_fixed(ni)
    } else {
        let permutation_slice = slice::from_raw_parts(permutation_ptr, ni);
        let permutation_vector: Vec<usize> =
            permutation_slice.iter().map(|x| *x as usize).collect();
        Permutation::from_vector(permutation_vector).unwrap()
    };
    let obj = JlpParameters::new(ni, Mass::new(mass), permutation).unwrap();
    // Then copy it to the heap (so we have a stable pointer to it).
    let boxed_obj = Box::new(obj);
    // Then return a pointer by converting our `Box<_>` into a raw pointer
    Box::into_raw(boxed_obj)
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__jlpparameters_free(obj: *mut JlpParameters) {
    // As a rule of thumb, freeing a null pointer is just a noop.
    if obj.is_null() {
        return;
    }
    // Convert the raw pointer back to a Box<_>
    let boxed = Box::from_raw(obj);
    // Then explicitly drop it (optional)
    drop(boxed);
}
