// Chinese restaurant process

use crate::clust::Clustering;
use crate::distr::{
    FullConditional, NormalizedProbabilityMassFunction, PartitionConditionalSampler,
    PartitionSampler, PredictiveProbabilityFunction, ProbabilityMassFunction,
};
use crate::perm::Permutation;
use crate::prelude::*;

use rand::Rng;
use statrs::function::gamma::ln_gamma;

#[derive(Debug, Clone)]
pub struct CrpParameters {
    n_items: usize,
    mass: Mass,
    discount: Discount,
}

impl CrpParameters {
    pub fn new_with_mass(n_items: usize, mass: Mass) -> Self {
        Self::new_with_mass_and_discount(n_items, mass, Discount::new(0.0))
    }

    pub fn new_with_mass_and_discount(n_items: usize, mass: Mass, discount: Discount) -> Self {
        Self {
            n_items,
            mass,
            discount,
        }
    }
}

impl PredictiveProbabilityFunction for CrpParameters {
    fn log_predictive_weight(
        &self,
        _item: usize,
        candidate_labels: &[usize],
        clustering: &Clustering,
    ) -> Vec<(usize, f64)> {
        if candidate_labels.len() == 1 {
            return vec![(candidate_labels[0], 0.0)];
        }
        let discount = self.discount.unwrap();
        candidate_labels
            .iter()
            .map(|label| {
                let size = clustering.size_of(*label);
                let value = if size == 0 {
                    self.mass.unwrap() + (clustering.n_clusters() as f64) * discount
                } else {
                    size as f64 - discount
                }
                .ln();
                (*label, value)
            })
            .collect()
    }
}

impl FullConditional for CrpParameters {
    fn log_full_conditional(&self, item: usize, clustering: &Clustering) -> Vec<(usize, f64)> {
        let discount = self.discount.unwrap();
        clustering
            .available_labels_for_reallocation(item)
            .map(|label| {
                let size = clustering.size_of_without(label, item);
                let value = if size == 0 {
                    self.mass.unwrap() + (clustering.n_clusters_without(item) as f64) * discount
                } else {
                    size as f64 - discount
                }
                .ln();
                (label, value)
            })
            .collect()
    }
}

impl PartitionSampler for CrpParameters {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering {
        crate::distr::default_partition_sampler_sample(
            self,
            &Permutation::natural_and_fixed(self.n_items),
            rng,
        )
    }
}

impl PartitionConditionalSampler for CrpParameters {
    fn sample_conditionally<T: Rng>(&self, clustering: Clustering, rng: &mut T) -> Clustering {
        crate::distr::default_partition_conditional_sampler_sample(
            self,
            &Permutation::natural_and_fixed(self.n_items),
            clustering,
            rng,
        )
    }
}

impl ProbabilityMassFunction for CrpParameters {
    fn log_pmf(&self, partition: &Clustering) -> f64 {
        let m = self.mass.unwrap();
        let d = self.discount.unwrap();
        let mut result = 0.0;
        if m > 0.0 {
            result -= ln_gamma(m + (partition.n_items_allocated() as f64)) - ln_gamma(m + 1.0)
        } else {
            for j in 1..partition.n_items_allocated() {
                result -= (m + j as f64).ln()
            }
        }
        if d == 0.0 {
            result += ((partition.n_clusters() - 1) as f64) * m.ln();
            for label in partition.active_labels() {
                result += ln_gamma(partition.size_of(*label) as f64);
            }
        } else {
            let mut cum_d = d;
            for _ in 1..partition.n_clusters() {
                result += (m + cum_d).ln();
                cum_d += d;
            }
            for label in partition.active_labels() {
                for i in 1..partition.size_of(*label) {
                    result += (i as f64 - d).ln();
                }
            }
        }
        result
    }
}

impl NormalizedProbabilityMassFunction for CrpParameters {}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_goodness_of_fit_constructive() {
        let parameters = CrpParameters::new_with_mass(5, Mass::new(2.0));
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
    fn test_pmf_without_discount() {
        let parameters = CrpParameters::new_with_mass(5, Mass::new(1.5));
        let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
        crate::testing::assert_pmf_sums_to_one(parameters.n_items, log_prob_closure, 0.0000001);
    }

    #[test]
    fn test_pmf_with_discount() {
        let parameters =
            CrpParameters::new_with_mass_and_discount(5, Mass::new(1.5), Discount::new(0.1));
        let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
        crate::testing::assert_pmf_sums_to_one(parameters.n_items, log_prob_closure, 0.0000001);
    }
}
