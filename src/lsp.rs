// Location scale partition distribution

use crate::clust::Clustering;
use crate::distr::{FullConditional, PartitionSampler, ProbabilityMassFunction};
use crate::perm::Permutation;
use crate::prelude::*;

use rand::prelude::*;
use rand_distr::Beta;
use rand_pcg::Pcg64Mcg;

#[derive(Debug, Clone)]
pub struct LspParameters {
    baseline_partition: Clustering,
    scale: Scale,
    rate: Rate,
    mass: Mass,
    permutation: Permutation,
}

impl LspParameters {
    pub fn new_with_scale(
        baseline: Clustering,
        scale: Scale,
        mass: Mass,
        permutation: Permutation,
    ) -> Option<Self> {
        if baseline.n_items() != permutation.n_items() {
            None
        } else {
            Some(Self {
                baseline_partition: baseline,
                scale,
                rate: Rate::new(1.0 / scale.unwrap()),
                mass,
                permutation,
            })
        }
    }
    pub fn new_with_rate(
        baseline: Clustering,
        rate: Rate,
        mass: Mass,
        permutation: Permutation,
    ) -> Option<Self> {
        if baseline.n_items() != permutation.n_items() {
            None
        } else {
            Some(Self {
                baseline_partition: baseline,
                scale: Scale::new(1.0 / rate.unwrap()),
                rate,
                mass,
                permutation,
            })
        }
    }

    pub fn shuffle_permutation<T: Rng>(&mut self, rng: &mut T) {
        self.permutation.shuffle(rng);
    }

    pub fn resample_rate<T: Rng>(&mut self, max: f64, shape1: f64, shape2: f64, rng: &mut T) {
        let beta = Beta::new(shape1, shape2).unwrap();
        self.rate = Rate::new(max * beta.sample(rng));
        self.scale = Scale::new(1.0 / self.rate.unwrap());
    }
}

impl FullConditional for LspParameters {
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

impl PartitionSampler for LspParameters {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering {
        engine(self, None, Some(rng)).0
    }
}

impl ProbabilityMassFunction for LspParameters {
    fn log_pmf(&self, partition: &Clustering) -> f64 {
        engine::<Pcg64Mcg>(self, Some(partition.allocation()), None).1
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
    let ni = parameters.baseline_partition.n_items();
    let mut log_probability = 0.0;
    let mut clustering = Clustering::unallocated(ni);
    let mut total_counter = vec![0.0; parameters.baseline_partition.max_label() + 1];
    let mut intersection_counter = Vec::with_capacity(total_counter.len());
    for _ in 0..total_counter.len() {
        intersection_counter.push(Vec::new())
    }
    let mut n_visited_subsets = 0;
    let mut visited_subsets_indicator = vec![false; total_counter.len()];
    for i in 0..clustering.n_items() {
        let ii = parameters.permutation.get(i);
        let baseline_subset_index = parameters.baseline_partition.get(ii);
        let n_occupied_subsets = clustering.n_clusters() as f64;
        let labels_and_weights = clustering
            .available_labels_for_allocation_with_target(target, ii)
            .map(|label| {
                let n_items_in_cluster = clustering.size_of(label);
                let weight = if n_items_in_cluster == 0 {
                    if n_occupied_subsets == 0.0 {
                        parameters.mass.unwrap()
                    } else {
                        {
                            if total_counter[baseline_subset_index] == 0.0 {
                                (parameters.mass.unwrap() + parameters.rate)
                                    / (parameters.mass.unwrap()
                                        + (n_visited_subsets as f64)
                                        + parameters.rate)
                            } else {
                                parameters.mass.unwrap()
                                    / (parameters.mass.unwrap()
                                        + (n_visited_subsets as f64)
                                        + parameters.rate)
                            }
                        }
                    }
                } else {
                    (1.0 + parameters.rate * intersection_counter[baseline_subset_index][label])
                        / (parameters.mass.unwrap()
                            + (n_visited_subsets as f64)
                            + parameters.rate * (n_items_in_cluster as f64))
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
            let parameters =
                LspParameters::new_with_rate(baseline, rate, Mass::new(1.0), permutation).unwrap();
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
        let mut rng = thread_rng();
        for baseline in Clustering::iter(n_items) {
            let baseline = Clustering::from_vector(baseline);
            let rate = Rate::new(rng.gen_range(0.0..10.0));
            let permutation = Permutation::random(n_items, &mut rng);
            let parameters =
                LspParameters::new_with_rate(baseline, rate, Mass::new(1.0), permutation).unwrap();
            let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
            crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
        }
    }
}
