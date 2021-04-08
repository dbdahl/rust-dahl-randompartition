// Chinese restaurant process

use crate::clust::Clustering;
use crate::distr::{
    FullConditional, PartitionSampler, PredictiveProbabilityFunction,
    PredictiveProbabilityFunctionOld,
};
use crate::prelude::*;
use crate::prior::PartitionLogProbability;

use crate::perm::Permutation;
use rand::Rng;
use statrs::function::gamma::ln_gamma;

#[derive(Debug, Clone)]
pub struct CrpParameters {
    mass: Mass,
    discount: Discount,
    n_items: usize,
}

impl CrpParameters {
    pub fn new_with_mass(mass: Mass, n_items: usize) -> Self {
        Self::new_with_mass_and_discount(mass, Discount::new(0.0), n_items)
    }

    pub fn new_with_mass_and_discount(mass: Mass, discount: Discount, n_items: usize) -> Self {
        Self {
            mass,
            discount,
            n_items,
        }
    }
}

impl PredictiveProbabilityFunctionOld for CrpParameters {
    fn log_predictive_probability(
        &self,
        item: usize,
        label: usize,
        clustering: &Clustering,
    ) -> f64 {
        let size = clustering.size_of_without(label, item);
        if size == 0 {
            self.mass.unwrap()
                + (clustering.n_clusters_without(item) as f64) * self.discount.unwrap()
        } else {
            size as f64 - self.discount.unwrap()
        }
        .ln()
    }
}

impl PredictiveProbabilityFunction for CrpParameters {
    fn log_predictive(&self, _item: usize, clustering: &Clustering) -> Vec<(usize, f64)> {
        let discount = self.discount.unwrap();
        clustering
            .available_labels_for_allocation()
            .map(|label| {
                let size = clustering.size_of(label);
                let value = if size == 0 {
                    self.mass.unwrap() + (clustering.n_clusters() as f64) * discount
                } else {
                    size as f64 - discount
                }
                .ln();
                (label, value)
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

/*
impl PartitionSampler for CrpParameters {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering {
        let mass = self.mass.unwrap();
        let discount = self.discount.unwrap();
        let mut clustering = Clustering::unallocated(self.n_items);
        clustering.allocate(0, 0);
        for i in 1..clustering.n_items() {
            let n_clusters = clustering.n_clusters();
            let weights = clustering.available_labels_for_allocation().map(|label| {
                let n_items_in_cluster = clustering.size_of(label);
                if n_items_in_cluster == 0 {
                    mass + (n_clusters as f64) * discount
                } else {
                    (n_items_in_cluster as f64) - discount
                }
            });
            // We're cheating here a bit in that we know the available labels are sequential from 0 to
            // clustering.n_clusters() + 1, exclusive.  This won't be the case in a Gibbs sampling
            // framework.
            use rand::distributions::{Distribution, WeightedIndex};
            let dist = WeightedIndex::new(weights).unwrap();
            clustering.allocate(i, dist.sample(rng));
        }
        clustering
    }
}
*/

impl PartitionLogProbability for CrpParameters {
    fn log_probability(&self, partition: &Clustering) -> f64 {
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
        let parameters = CrpParameters::new_with_mass(Mass::new(2.0), 5);
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
        let parameters =
            CrpParameters::new_with_mass_and_discount(Mass::new(2.0), Discount::new(0.1), 5);
        let l = |_i: usize, _indices: &[usize]| 0.0;
        let mut clustering = Clustering::one_cluster(parameters.n_items);
        let rng = &mut thread_rng();
        let permutation = Permutation::random(clustering.n_items(), rng);
        let sample_closure = || {
            clustering = crate::mcmc::update_neal_algorithm3_v2(
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
    fn test_pmf_without_discount() {
        let parameters = CrpParameters::new_with_mass(Mass::new(1.5), 5);
        let log_prob_closure = |clustering: &mut Clustering| parameters.log_probability(clustering);
        crate::testing::assert_pmf_sums_to_one(parameters.n_items, log_prob_closure, 0.0000001);
    }

    #[test]
    fn test_pmf_with_discount() {
        let parameters =
            CrpParameters::new_with_mass_and_discount(Mass::new(1.5), Discount::new(0.1), 5);
        let log_prob_closure = |clustering: &mut Clustering| parameters.log_probability(clustering);
        crate::testing::assert_pmf_sums_to_one(parameters.n_items, log_prob_closure, 0.0000001);
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__crpparameters_new(
    n_items: i32,
    mass: f64,
    discount: f64,
) -> *mut CrpParameters {
    let d = Discount::new(discount);
    let m = Mass::new_with_variable_constraint(mass, discount);
    // First we create a new object.
    let obj = CrpParameters::new_with_mass_and_discount(m, d, n_items as usize);
    // Then copy it to the heap (so we have a stable pointer to it).
    let boxed_obj = Box::new(obj);
    // Then return a pointer by converting our `Box<_>` into a raw pointer
    Box::into_raw(boxed_obj)
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__crpparameters_free(obj: *mut CrpParameters) {
    // As a rule of thumb, freeing a null pointer is just a noop.
    if obj.is_null() {
        return;
    }
    // Convert the raw pointer back to a Box<_>
    let boxed = Box::from_raw(obj);
    // Then explicitly drop it (optional)
    drop(boxed);
}
