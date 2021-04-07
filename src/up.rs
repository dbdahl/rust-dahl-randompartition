// Chinese restaurant process

use crate::clust::Clustering;
use crate::distr::{PartitionSampler, PredictiveProbabilityFunction};
use crate::prior::PartitionLogProbability;

use dahl_bellnumber::UniformDistributionCache;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct UpParameters {
    pub n_items: usize,
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
    fn log_predictive_probability(
        &self,
        item: usize,
        label: usize,
        clustering: &Clustering,
    ) -> f64 {
        let n_allocated = clustering.n_items_allocated_without(item);
        let n_clusters = clustering.n_clusters_without(item);
        let (left, right) = self
            .cache
            .probs_for_uniform(self.n_items - n_allocated, n_clusters);
        let size = clustering.size_of_without(label, item);
        let predictive_probability = if size == 0 { right } else { left };
        predictive_probability.ln()
    }
}

impl PartitionSampler for UpParameters {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering {
        crate::distr::default_partition_sampler_sample_without_permutation(self, self.n_items, rng)
    }
}

impl PartitionLogProbability for UpParameters {
    fn log_probability(&self, _partition: &Clustering) -> f64 {
        -self.cache.lbell(self.n_items)
    }

    fn is_normalized(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perm::Permutation;
    use rand::prelude::*;

    #[test]
    fn test_goodness_of_fit_constructive() {
        let parameters = UpParameters::new(5);
        let sample_closure = || parameters.sample(&mut thread_rng());
        let log_prob_closure = |clustering: &mut Clustering| parameters.log_probability(clustering);
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
    fn test_goodness_of_fit_neal_algorithm3() {
        let parameters = UpParameters::new(5);
        let l = |_i: usize, _indices: &[usize]| 0.0;
        let mut clustering = Clustering::one_cluster(parameters.n_items);
        let rng = &mut thread_rng();
        let permutation = Permutation::random(clustering.n_items(), rng);
        let sample_closure = || {
            clustering = crate::mcmc::update_neal_algorithm3(
                1,
                &clustering,
                &permutation,
                &parameters,
                &l,
                &mut thread_rng(),
            );
            clustering.relabel(0, None, false).0
        };
        let log_prob_closure = |clustering: &mut Clustering| parameters.log_probability(clustering);
        crate::testing::assert_goodness_of_fit(
            10000,
            parameters.n_items,
            sample_closure,
            log_prob_closure,
            5,
            0.001,
        );
    }

    #[test]
    fn test_pmf() {
        let parameters = UpParameters::new(5);
        let log_prob_closure = |clustering: &mut Clustering| parameters.log_probability(clustering);
        crate::testing::assert_pmf_sums_to_one(parameters.n_items, log_prob_closure, 0.0000001);
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__urpparameters_new(
    n_items: i32,
) -> *mut UpParameters {
    // First we create a new object.
    let obj = UpParameters::new(n_items as usize);
    // Then copy it to the heap (so we have a stable pointer to it).
    let boxed_obj = Box::new(obj);
    // Then return a pointer by converting our `Box<_>` into a raw pointer
    Box::into_raw(boxed_obj)
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__urpparameters_free(obj: *mut UpParameters) {
    // As a rule of thumb, freeing a null pointer is just a noop.
    if obj.is_null() {
        return;
    }
    // Convert the raw pointer back to a Box<_>
    let boxed = Box::from_raw(obj);
    // Then explicitly drop it (optional)
    drop(boxed);
}
