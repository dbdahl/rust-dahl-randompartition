// Chinese restaurant process

use crate::clust::Clustering;
use crate::distr::{
    FullConditional, PartitionSampler, PredictiveProbabilityFunction, ProbabilityMassFunction,
};

use crate::perm::Permutation;
use dahl_bellnumber::UniformDistributionCache;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct UpParameters {
    n_items: usize,
    cache: UniformDistributionCache,
}

impl UpParameters {
    pub fn new(n_items: usize) -> Self {
        Self {
            n_items,
            cache: UniformDistributionCache::new(n_items),
        }
    }
}

impl PredictiveProbabilityFunction for UpParameters {
    fn log_predictive(
        &self,
        _item: usize,
        candidate_labels: Vec<usize>,
        clustering: &Clustering,
    ) -> Vec<(usize, f64)> {
        let n_allocated = clustering.n_items_allocated();
        let n_clusters = clustering.n_clusters();
        let (left, right) = {
            let x = self
                .cache
                .probs_for_uniform(self.n_items - n_allocated, n_clusters);
            (x.0.ln(), x.1.ln())
        };
        candidate_labels
            .into_iter()
            .map(|label| {
                (
                    label,
                    if clustering.size_of(label) == 0 {
                        right
                    } else {
                        left
                    },
                )
            })
            .collect()
    }
}

impl FullConditional for UpParameters {
    fn log_full_conditional<'a>(
        &'a self,
        item: usize,
        clustering: &'a Clustering,
    ) -> Vec<(usize, f64)> {
        clustering
            .available_labels_for_reallocation(item)
            .map(|label| (label, 0.0))
            .collect()
    }
}

impl PartitionSampler for UpParameters {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering {
        crate::distr::default_partition_sampler_sample(
            self,
            &Permutation::natural_and_fixed(self.n_items),
            rng,
        )
    }
}

impl ProbabilityMassFunction for UpParameters {
    fn log_pmf(&self, _partition: &Clustering) -> f64 {
        -self.cache.lbell(self.n_items)
    }
    fn is_normalized(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_goodness_of_fit_constructive() {
        let parameters = UpParameters::new(5);
        let sample_closure = || parameters.sample(&mut thread_rng());
        let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
        crate::testing::assert_goodness_of_fit(
            100000,
            parameters.n_items,
            sample_closure,
            log_prob_closure,
            1,
            0.001,
        );
    }

    #[test]
    fn test_pmf() {
        let parameters = UpParameters::new(5);
        let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
        crate::testing::assert_pmf_sums_to_one(parameters.n_items, log_prob_closure, 0.0000001);
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__upparameters_new(n_items: i32) -> *mut UpParameters {
    // First we create a new object.
    let obj = UpParameters::new(n_items as usize);
    // Then copy it to the heap (so we have a stable pointer to it).
    let boxed_obj = Box::new(obj);
    // Then return a pointer by converting our `Box<_>` into a raw pointer
    Box::into_raw(boxed_obj)
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__upparameters_free(obj: *mut UpParameters) {
    // As a rule of thumb, freeing a null pointer is just a noop.
    if obj.is_null() {
        return;
    }
    // Convert the raw pointer back to a Box<_>
    let boxed = Box::from_raw(obj);
    // Then explicitly drop it (optional)
    drop(boxed);
}
