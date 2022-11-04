// Focal random partition distribution

use crate::clust::Clustering;
use crate::distr::{
    FullConditional, HasPermutation, HasVectorShrinkage, NormalizedProbabilityMassFunction,
    PartitionSampler, ProbabilityMassFunction,
};
use crate::perm::Permutation;
use crate::prelude::*;
use crate::shrink::Shrinkage;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;

#[derive(Debug, Clone)]
pub struct FrpParameters {
    baseline_partition: Clustering,
    pub shrinkage: Shrinkage,
    pub permutation: Permutation,
    mass: Mass,
    discount: Discount,
    power: Power,
}

impl FrpParameters {
    pub fn new(
        baseline: Clustering,
        shrinkage: Shrinkage,
        permutation: Permutation,
        mass: Mass,
        discount: Discount,
        power: Power,
    ) -> Option<Self> {
        if (shrinkage.n_items() != baseline.n_items())
            || (baseline.n_items() != permutation.n_items())
        {
            None
        } else {
            Some(Self {
                baseline_partition: baseline.standardize(),
                shrinkage,
                permutation,
                mass,
                discount,
                power,
            })
        }
    }
}

impl FullConditional for FrpParameters {
    fn log_full_conditional<'a>(
        &'a self,
        item: usize,
        clustering: &'a Clustering,
    ) -> Vec<(usize, f64)> {
        let mut p = clustering.allocation().clone();
        clustering
            .available_labels_for_reallocation(item)
            .map(|label| {
                p[item] = label;
                (label, engine::<Pcg64Mcg>(self, Some(&p[..]), None).1)
            })
            .collect()
    }
}

impl PartitionSampler for FrpParameters {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering {
        engine(self, None, Some(rng)).0
    }
}

impl ProbabilityMassFunction for FrpParameters {
    fn log_pmf(&self, partition: &Clustering) -> f64 {
        engine::<Pcg64Mcg>(self, Some(partition.allocation()), None).1
    }
}

impl NormalizedProbabilityMassFunction for FrpParameters {}

impl HasPermutation for FrpParameters {
    fn permutation(&self) -> &Permutation {
        &self.permutation
    }
    fn permutation_mut(&mut self) -> &mut Permutation {
        &mut self.permutation
    }
}

impl HasVectorShrinkage for FrpParameters {
    fn shrinkage(&self) -> &Shrinkage {
        &self.shrinkage
    }
    fn shrinkage_mut(&mut self) -> &mut Shrinkage {
        &mut self.shrinkage
    }
}

fn engine<T: Rng>(
    parameters: &FrpParameters,
    target: Option<&[usize]>,
    mut rng: Option<&mut T>,
) -> (Clustering, f64) {
    let ni = parameters.baseline_partition.n_items();
    let mass = parameters.mass.unwrap();
    let discount = parameters.discount.unwrap();
    let power = parameters.power.unwrap();
    let mut log_probability = 0.0;
    let mut clustering = Clustering::unallocated(ni);
    let mut total_counter = vec![0.0; parameters.baseline_partition.max_label() + 1];
    let mut intersection_counter = Vec::with_capacity(total_counter.len());
    for _ in 0..total_counter.len() {
        intersection_counter.push(Vec::new())
    }
    for i in 0..clustering.n_items() {
        let ii = parameters.permutation.get(i);
        let baseline_subset_index = parameters.baseline_partition.get(ii);
        let scaled_shrinkage = (i as f64) * parameters.shrinkage[ii];
        let normalized_scaled_shrinkage = if total_counter[baseline_subset_index] == 0.0 {
            0.0
        } else {
            scaled_shrinkage / total_counter[baseline_subset_index]
        };
        let n_occupied_subsets = clustering.n_clusters() as f64;
        let labels_and_weights = clustering
            .available_labels_for_allocation_with_target(target, ii)
            .map(|label| {
                let n_items_in_cluster = clustering.size_of(label);
                let weight = if n_items_in_cluster == 0 {
                    if n_occupied_subsets == 0.0 {
                        1.0
                    } else {
                        mass + discount * n_occupied_subsets + {
                            if total_counter[baseline_subset_index] == 0.0 {
                                scaled_shrinkage
                            } else {
                                0.0
                            }
                        }
                    }
                } else {
                    ((n_items_in_cluster as f64) - discount).powf(power)
                        + normalized_scaled_shrinkage
                            * intersection_counter[baseline_subset_index][label]
                };
                (label, weight)
            });
        let (subset_index, log_probability_contribution) = match &mut rng {
            Some(r) => clustering.select(labels_and_weights, false, 0, Some(r), true),
            None => clustering.select::<Pcg64Mcg, _>(
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
    }
    (clustering, log_probability)
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
        let power = Power::new(0.5);
        let mut rng = thread_rng();
        for baseline in Clustering::iter(n_items) {
            let baseline = Clustering::from_vector(baseline);
            let mut vec = Vec::with_capacity(baseline.n_clusters());
            for _ in 0..baseline.n_items() {
                vec.push(rng.gen_range(0.0..10.0));
            }
            let shrinkage = Shrinkage::from(&vec[..]).unwrap();
            let permutation = Permutation::random(n_items, &mut rng);
            let parameters =
                FrpParameters::new(baseline, shrinkage, permutation, mass, discount, power)
                    .unwrap();
            let sample_closure = || parameters.sample(&mut thread_rng());
            let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
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
        let discount = 0.1;
        let mass = Mass::new_with_variable_constraint(2.0, discount);
        let discount = Discount::new(discount);
        let power = Power::new(0.5);
        let mut rng = thread_rng();
        for baseline in Clustering::iter(n_items) {
            let baseline = Clustering::from_vector(baseline);
            let mut vec = Vec::with_capacity(baseline.n_clusters());
            for _ in 0..baseline.n_items() {
                vec.push(rng.gen_range(0.0..10.0));
            }
            let shrinkage = Shrinkage::from(&vec[..]).unwrap();
            let permutation = Permutation::random(n_items, &mut rng);
            let parameters =
                FrpParameters::new(baseline, shrinkage, permutation, mass, discount, power)
                    .unwrap();
            let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
            crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
        }
    }
}
