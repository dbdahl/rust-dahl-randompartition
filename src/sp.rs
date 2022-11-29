// Shrinkage partition distribution

use crate::clust::Clustering;
use crate::distr::{
    FullConditional, HasPermutation, HasVectorShrinkage, NormalizedProbabilityMassFunction,
    PartitionSampler, PredictiveProbabilityFunction, ProbabilityMassFunction,
};
use crate::perm::Permutation;
use crate::shrink::Shrinkage;

use rand::prelude::*;
use rand_pcg::Pcg64Mcg;

#[derive(Debug, Clone)]
pub struct SpParameters<D: PredictiveProbabilityFunction + Clone> {
    pub baseline_partition: Clustering,
    pub shrinkage: Shrinkage,
    pub permutation: Permutation,
    baseline_ppf: D,
}

impl<D: PredictiveProbabilityFunction + Clone> SpParameters<D> {
    pub fn new(
        baseline_partition: Clustering,
        shrinkage: Shrinkage,
        permutation: Permutation,
        baseline_ppf: D,
    ) -> Option<Self> {
        if (shrinkage.n_items() != baseline_partition.n_items())
            || (baseline_partition.n_items() != permutation.n_items())
        {
            None
        } else {
            Some(Self {
                baseline_partition: baseline_partition.standardize(),
                shrinkage,
                permutation,
                baseline_ppf,
            })
        }
    }
}

fn use_slow_implementation() -> bool {
    match std::env::var("DBD_SP_SLOW") {
        Ok(val) => val != "",
        Err(_) => false,
    }
}

fn expand_counts(counts: &mut Vec<Vec<usize>>, new_len: usize) {
    counts.iter_mut().map(|x| x.resize(new_len, 0)).collect()
}

impl<D: PredictiveProbabilityFunction + Clone> FullConditional for SpParameters<D> {
    // Implement starting only at item and subsequent items.
    fn log_full_conditional(&self, item: usize, clustering: &Clustering) -> Vec<(usize, f64)> {
        if use_slow_implementation() {
            let mut target = clustering.allocation().clone();
            let candidate_labels = clustering.available_labels_for_reallocation(item);
            let mut partial_clustering = clustering.clone();
            for i in self.permutation.n_items_before(item)..partial_clustering.n_items() {
                partial_clustering.remove(self.permutation.get(i));
            }
            candidate_labels
                .map(|label| {
                    target[item] = label;
                    (
                        label,
                        engine_slow_implementation::<D, Pcg64Mcg>(
                            self,
                            partial_clustering.clone(),
                            Some(&target[..]),
                            None,
                        )
                        .1,
                    )
                })
                .collect()
        } else {
            let mut target = clustering.allocation().clone();
            let candidate_labels = clustering.available_labels_for_reallocation(item);
            let mut partial_clustering = clustering.clone();
            for i in self.permutation.n_items_before(item)..partial_clustering.n_items() {
                partial_clustering.remove(self.permutation.get(i));
            }
            let (mut counts_marginal, mut counts) = {
                let m = self.baseline_partition.max_label() + 1;
                (vec![0; m], vec![Vec::new(); m])
            };
            let max_label = partial_clustering.max_label();
            if max_label >= counts[0].len() {
                expand_counts(&mut counts, partial_clustering.max_label() + 1)
            }
            for i in 0..partial_clustering.n_items_allocated() {
                let item = self.permutation.get(i);
                let label_in_baseline = self.baseline_partition.get(item);
                let label = target[item];
                counts_marginal[label_in_baseline] += 1;
                counts[label_in_baseline][label] += 1;
            }
            candidate_labels
                .map(|label| {
                    target[item] = label;
                    (
                        label,
                        engine::<D, Pcg64Mcg>(
                            self,
                            partial_clustering.clone(),
                            counts_marginal.clone(),
                            counts.clone(),
                            Some(&target[..]),
                            None,
                        )
                        .1,
                    )
                })
                .collect()
        }
    }
}

impl<D: PredictiveProbabilityFunction + Clone> PartitionSampler for SpParameters<D> {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering {
        engine_full(self, None, Some(rng)).0
    }
}

impl<D: PredictiveProbabilityFunction + Clone> ProbabilityMassFunction for SpParameters<D> {
    fn log_pmf(&self, partition: &Clustering) -> f64 {
        engine_full::<D, Pcg64Mcg>(self, Some(partition.allocation()), None).1
    }
}

impl<D: PredictiveProbabilityFunction + Clone> NormalizedProbabilityMassFunction
    for SpParameters<D>
{
}

impl<D: PredictiveProbabilityFunction + Clone> HasPermutation for SpParameters<D> {
    fn permutation(&self) -> &Permutation {
        &self.permutation
    }
    fn permutation_mut(&mut self) -> &mut Permutation {
        &mut self.permutation
    }
}

impl<D: PredictiveProbabilityFunction + Clone> HasVectorShrinkage for SpParameters<D> {
    fn shrinkage(&self) -> &Shrinkage {
        &self.shrinkage
    }
    fn shrinkage_mut(&mut self) -> &mut Shrinkage {
        &mut self.shrinkage
    }
}

fn engine_full<'a, D: PredictiveProbabilityFunction + Clone, T: Rng>(
    parameters: &'a SpParameters<D>,
    target: Option<&[usize]>,
    rng: Option<&mut T>,
) -> (Clustering, f64) {
    if use_slow_implementation() {
        engine_slow_implementation(
            parameters,
            Clustering::unallocated(parameters.baseline_partition.n_items()),
            target,
            rng,
        )
    } else {
        let m = parameters.baseline_partition.max_label() + 1;
        engine(
            parameters,
            Clustering::unallocated(parameters.baseline_partition.n_items()),
            vec![0; m],
            vec![Vec::new(); m],
            target,
            rng,
        )
    }
}

fn log_weights_to_probabilities<T>(x: &mut Vec<(T, f64)>) {
    let max_weight = x.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
    for (_, w) in x.iter_mut() {
        *w = (*w - max_weight).exp()
    }
    let sum = x.iter().fold(0.0, |acc, z| acc + z.1);
    for y in x.iter_mut().map(|z| &mut z.1) {
        *y /= sum;
    }
}

fn engine_slow_implementation<'a, D: PredictiveProbabilityFunction + Clone, T: Rng>(
    parameters: &'a SpParameters<D>,
    mut clustering: Clustering,
    target: Option<&[usize]>,
    mut rng: Option<&mut T>,
) -> (Clustering, f64) {
    let mut log_probability = 0.0;
    for i in clustering.n_items_allocated()..clustering.n_items() {
        let item = parameters.permutation.get(i);
        let weight_on_baseline_ppf = 1.0 / (1.0 + parameters.shrinkage[item]);
        let weight_on_shrinkage_ppf = 1.0 - weight_on_baseline_ppf;
        let candidate_labels: Vec<usize> = clustering
            .available_labels_for_allocation_with_target(target, item)
            .collect();
        let baseline_ppf = {
            let mut x =
                parameters
                    .baseline_ppf
                    .log_predictive_weight(item, &candidate_labels, &clustering);
            log_weights_to_probabilities(&mut x);
            x
        };
        let label_in_baseline = parameters.baseline_partition.get(item);
        let mut items_in_baseline = parameters.baseline_partition.items_of(label_in_baseline);
        items_in_baseline.sort_unstable();
        let mut new_index = 0;
        let mut denominator = 0.0;
        let mut shrinkage_ppf: Vec<_> = baseline_ppf
            .iter()
            .enumerate()
            .map(|(index, &(label, _))| {
                let items = clustering.items_of(label);
                if items.is_empty() {
                    new_index = index;
                    return 0.0;
                }
                let intersection = items
                    .iter()
                    .filter(|x| items_in_baseline.binary_search(x).is_ok());
                let result = intersection.count() as f64;
                denominator += result;
                result
            })
            .collect();
        if denominator == 0.0 {
            shrinkage_ppf[new_index] = 1.0;
        } else {
            for v in shrinkage_ppf.iter_mut() {
                *v /= denominator;
            }
        }
        let labels_and_probabilities: Vec<_> = baseline_ppf
            .into_iter()
            .zip(shrinkage_ppf)
            .map(|((label, p1), p2)| {
                (
                    label,
                    weight_on_baseline_ppf * p1 + weight_on_shrinkage_ppf * p2,
                )
            })
            .collect();
        let (label, log_probability_contribution) = match &mut rng {
            Some(r) => clustering.select(
                labels_and_probabilities.into_iter(),
                false,
                true,
                0,
                Some(r),
                true,
            ),
            None => clustering.select::<Pcg64Mcg, _>(
                labels_and_probabilities.into_iter(),
                false,
                true,
                target.unwrap()[item],
                None,
                true,
            ),
        };
        log_probability += log_probability_contribution;
        clustering.allocate(item, label);
    }
    (clustering, log_probability)
}

fn engine<'a, D: PredictiveProbabilityFunction + Clone, T: Rng>(
    parameters: &'a SpParameters<D>,
    mut clustering: Clustering,
    mut counts_marginal: Vec<usize>,
    mut counts: Vec<Vec<usize>>,
    target: Option<&[usize]>,
    mut rng: Option<&mut T>,
) -> (Clustering, f64) {
    let mut log_probability = 0.0;
    for i in clustering.n_items_allocated()..clustering.n_items() {
        let item = parameters.permutation.get(i);
        let label_in_baseline = parameters.baseline_partition.get(item);
        let shrinkage = parameters.shrinkage[item];
        let candidate_labels: Vec<usize> = clustering
            .available_labels_for_allocation_with_target(target, item)
            .collect();
        let max_candidate_label = *candidate_labels.iter().max().unwrap();
        if max_candidate_label >= counts[label_in_baseline].len() {
            expand_counts(&mut counts, max_candidate_label + 1)
        }
        let n_marginal = counts_marginal[label_in_baseline] as f64;
        let multiplier = shrinkage / n_marginal;
        let labels_and_log_weights = parameters
            .baseline_ppf
            .log_predictive_weight(item, &candidate_labels, &clustering)
            .into_iter()
            .map(|(label, log_probability)| {
                let n_joint = counts[label_in_baseline][label] as f64;
                let lp = log_probability
                    + if n_joint > 0.0 {
                        multiplier * n_joint
                    } else {
                        if n_marginal == 0.0 && clustering.size_of(label) == 0 {
                            shrinkage
                        } else {
                            0.0
                        }
                    };
                (label, lp)
            });
        let (label, log_probability_contribution) = match &mut rng {
            Some(r) => clustering.select(labels_and_log_weights, true, false, 0, Some(r), true),
            None => clustering.select::<Pcg64Mcg, _>(
                labels_and_log_weights,
                true,
                false,
                target.unwrap()[item],
                None,
                true,
            ),
        };
        log_probability += log_probability_contribution;
        clustering.allocate(item, label);
        counts_marginal[label_in_baseline] += 1;
        counts[label_in_baseline][label] += 1;
    }
    (clustering, log_probability)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crp::CrpParameters;
    use crate::prelude::*;

    #[test]
    fn test_normalize_log_weights() {
        let mut y = vec![(0, 1.0), (1, 4.0), (2, 5.0)];
        let sum: f64 = y.iter().map(|x| x.1).sum();
        y.iter_mut().map(|x| &mut x.1).for_each(|yy| {
            *yy /= sum;
        });
        let mut x = y.clone();
        x.iter_mut()
            .map(|y| &mut y.1)
            .for_each(|yy| *yy = (*yy).ln() + 23.0);
        log_weights_to_probabilities(&mut x);
        let epsilon = 0.0001;
        for (xx, yy) in y.iter().zip(x).map(|(x, y)| (x.1, y.1)) {
            assert!((xx - yy).abs() < epsilon);
        }
    }

    #[test]
    fn test_goodness_of_fit_constructive() {
        let n_items = 4;
        let discount = 0.1;
        let mass = Mass::new_with_variable_constraint(2.0, discount);
        let discount = Discount::new(discount);
        let mut rng = thread_rng();
        for target in Clustering::iter(n_items) {
            let target = Clustering::from_vector(target);
            let mut vec = Vec::with_capacity(target.n_clusters());
            for _ in 0..target.n_items() {
                vec.push(rng.gen_range(0.0..10.0));
            }
            let shrinkage = Shrinkage::from(&vec[..]).unwrap();
            let permutation = Permutation::random(n_items, &mut rng);
            let baseline_distribution =
                CrpParameters::new_with_mass_and_discount(n_items, mass, discount);
            let parameters =
                SpParameters::new(target, shrinkage, permutation, baseline_distribution).unwrap();
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
        let mut rng = thread_rng();
        for target in Clustering::iter(n_items) {
            let target = Clustering::from_vector(target);
            let mut vec = Vec::with_capacity(target.n_clusters());
            for _ in 0..target.n_items() {
                vec.push(rng.gen_range(0.0..10.0));
            }
            let shrinkage = Shrinkage::from(&vec[..]).unwrap();
            let permutation = Permutation::random(n_items, &mut rng);
            let baseline_distribution =
                CrpParameters::new_with_mass_and_discount(n_items, mass, discount);
            let parameters =
                SpParameters::new(target, shrinkage, permutation, baseline_distribution).unwrap();
            let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
            crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
        }
    }
}
