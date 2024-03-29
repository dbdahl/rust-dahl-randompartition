// Uniform partition distribution

use crate::clust::Clustering;
use crate::distr::{
    FullConditional, HasPermutation, NormalizedProbabilityMassFunction, PartitionSampler,
    PredictiveProbabilityFunction, ProbabilityMassFunction,
};
use crate::perm::Permutation;
use crate::prelude::Concentration;

use rand::Rng;

#[derive(Debug, Clone)]
pub struct JlpParameters {
    n_items: usize,
    concentration: Concentration,
    pub permutation: Permutation,
}

impl JlpParameters {
    pub fn new(
        n_items: usize,
        concentration: Concentration,
        permutation: Permutation,
    ) -> Option<Self> {
        if n_items != permutation.n_items() {
            None
        } else {
            Some(Self {
                n_items,
                concentration,
                permutation,
            })
        }
    }
}

impl PredictiveProbabilityFunction for JlpParameters {
    fn log_predictive_weight(
        &self,
        _item: usize,
        candidate_labels: &[usize],
        clustering: &Clustering,
    ) -> Vec<(usize, f64)> {
        candidate_labels
            .iter()
            .map(|label| {
                (
                    *label,
                    if clustering.size_of(*label) == 0 {
                        self.concentration.ln()
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
}

impl HasPermutation for JlpParameters {
    fn permutation(&self) -> &Permutation {
        &self.permutation
    }
    fn permutation_mut(&mut self) -> &mut Permutation {
        &mut self.permutation
    }
}

impl NormalizedProbabilityMassFunction for JlpParameters {}

mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_pmf() {
        let parameters = JlpParameters::new(
            5,
            Concentration::new(2.0).unwrap(),
            Permutation::natural_and_fixed(5),
        )
        .unwrap();
        let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
        crate::testing::assert_pmf_sums_to_one(parameters.n_items, log_prob_closure, 0.0000001);
    }
}
