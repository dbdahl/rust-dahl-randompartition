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
pub struct Old2SpParameters<D: PredictiveProbabilityFunction + Clone> {
    pub anchor: Clustering,
    pub shrinkage: Shrinkage,
    pub permutation: Permutation,
    baseline_ppf: D,
}

impl<D: PredictiveProbabilityFunction + Clone> Old2SpParameters<D> {
    pub fn new(
        anchor: Clustering,
        shrinkage: Shrinkage,
        permutation: Permutation,
        baseline_ppf: D,
    ) -> Option<Self> {
        if (shrinkage.n_items() != anchor.n_items())
            || (anchor.n_items() != permutation.n_items())
        {
            None
        } else {
            Some(Self {
                anchor: anchor.standardize(),
                shrinkage,
                permutation,
                baseline_ppf,
            })
        }
    }
}

fn expand_counts(counts: &mut [Vec<usize>], new_len: usize) {
    counts.iter_mut().map(|x| x.resize(new_len, 0)).collect()
}

impl<D: PredictiveProbabilityFunction + Clone> FullConditional for Old2SpParameters<D> {
    // Implement starting only at item and subsequent items.
    fn log_full_conditional(&self, item: usize, clustering: &Clustering) -> Vec<(usize, f64)> {
        let mut target = clustering.allocation().clone();
        let candidate_labels = clustering.available_labels_for_reallocation(item);
        let mut partial_clustering = clustering.clone();
        for i in self.permutation.n_items_before(item)..partial_clustering.n_items() {
            partial_clustering.remove(self.permutation.get(i));
        }
        let (mut counts_marginal, mut counts) = {
            let m = self.anchor.max_label() + 1;
            (vec![0; m], vec![Vec::new(); m])
        };
        let max_label = partial_clustering.max_label();
        if max_label >= counts[0].len() {
            expand_counts(&mut counts, partial_clustering.max_label() + 1)
        }
        for i in 0..partial_clustering.n_items_allocated() {
            let item = self.permutation.get(i);
            let label_in_baseline = self.anchor.get(item);
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

impl<D: PredictiveProbabilityFunction + Clone> PartitionSampler for Old2SpParameters<D> {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering {
        engine_full(self, None, Some(rng)).0
    }
}

impl<D: PredictiveProbabilityFunction + Clone> ProbabilityMassFunction for Old2SpParameters<D> {
    fn log_pmf(&self, partition: &Clustering) -> f64 {
        engine_full::<D, Pcg64Mcg>(self, Some(partition.allocation()), None).1
    }
}

impl<D: PredictiveProbabilityFunction + Clone> NormalizedProbabilityMassFunction
    for Old2SpParameters<D>
{
}

impl<D: PredictiveProbabilityFunction + Clone> HasPermutation for Old2SpParameters<D> {
    fn permutation(&self) -> &Permutation {
        &self.permutation
    }
    fn permutation_mut(&mut self) -> &mut Permutation {
        &mut self.permutation
    }
}

impl<D: PredictiveProbabilityFunction + Clone> HasVectorShrinkage for Old2SpParameters<D> {
    fn shrinkage(&self) -> &Shrinkage {
        &self.shrinkage
    }
    fn shrinkage_mut(&mut self) -> &mut Shrinkage {
        &mut self.shrinkage
    }
}

fn engine_full<D: PredictiveProbabilityFunction + Clone, T: Rng>(
    parameters: &Old2SpParameters<D>,
    target: Option<&[usize]>,
    rng: Option<&mut T>,
) -> (Clustering, f64) {
    let m = parameters.anchor.max_label() + 1;
    engine(
        parameters,
        Clustering::unallocated(parameters.anchor.n_items()),
        vec![0; m],
        vec![Vec::new(); m],
        target,
        rng,
    )
}

fn engine<D: PredictiveProbabilityFunction + Clone, T: Rng>(
    parameters: &Old2SpParameters<D>,
    mut clustering: Clustering,
    mut counts_marginal: Vec<usize>,
    mut counts: Vec<Vec<usize>>,
    target: Option<&[usize]>,
    mut rng: Option<&mut T>,
) -> (Clustering, f64) {
    let mut log_probability = 0.0;
    for i in clustering.n_items_allocated()..clustering.n_items() {
        let item = parameters.permutation.get(i);
        let label_in_baseline = parameters.anchor.get(item);
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
                    } else if n_marginal == 0.0 && clustering.size_of(label) == 0 {
                        shrinkage
                    } else {
                        0.0
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
            let baseline =
                CrpParameters::new_with_mass_and_discount(n_items, mass, discount);
            let parameters =
                Old2SpParameters::new(target, shrinkage, permutation, baseline)
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
        let mut rng = thread_rng();
        for target in Clustering::iter(n_items) {
            let target = Clustering::from_vector(target);
            let mut vec = Vec::with_capacity(target.n_clusters());
            for _ in 0..target.n_items() {
                vec.push(rng.gen_range(0.0..10.0));
            }
            let shrinkage = Shrinkage::from(&vec[..]).unwrap();
            let permutation = Permutation::random(n_items, &mut rng);
            let baseline =
                CrpParameters::new_with_mass_and_discount(n_items, mass, discount);
            let parameters =
                Old2SpParameters::new(target, shrinkage, permutation, baseline)
                    .unwrap();
            let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
            crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
        }
    }
}
