// Shrinkage partition distribution

use crate::clust::Clustering;
use crate::distr::{
    FullConditional, HasGrit, HasPermutation, HasVectorShrinkage,
    NormalizedProbabilityMassFunction, PartitionSampler, PredictiveProbabilityFunction,
    ProbabilityMassFunction, ProbabilityMassFunctionPartial,
};
use crate::perm::Permutation;
use crate::prelude::*;
use crate::shrink::Shrinkage;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;

#[derive(Debug, Clone)]
pub struct SpParameters<D: PredictiveProbabilityFunction + Clone> {
    pub anchor: Clustering,
    pub shrinkage: Shrinkage,
    pub permutation: Permutation,
    pub grit: Grit,
    pub baseline_ppf: D,
}

impl<D: PredictiveProbabilityFunction + Clone> SpParameters<D> {
    pub fn new(
        anchor: Clustering,
        shrinkage: Shrinkage,
        permutation: Permutation,
        grit: Grit,
        baseline_ppf: D,
    ) -> Option<Self> {
        if (shrinkage.n_items() != anchor.n_items()) || (anchor.n_items() != permutation.n_items())
        {
            None
        } else {
            Some(Self {
                anchor: anchor.standardize(),
                shrinkage,
                permutation,
                grit,
                baseline_ppf,
            })
        }
    }

    fn prepare_for_partial(
        &self,
        item: usize,
        clustering: &Clustering,
    ) -> (Clustering, Vec<f64>, Vec<Vec<f64>>) {
        let mut partial_clustering = clustering.clone();
        for i in self.permutation.n_items_before(item)..partial_clustering.n_items() {
            partial_clustering.remove(self.permutation.get(i));
        }
        let (mut counts_marginal, mut counts_joint) = {
            let m = self.anchor.max_label() + 1;
            (Vec::new(), vec![Vec::new(); m])
        };
        let max_label = partial_clustering.max_label();
        if max_label >= counts_marginal.len() {
            expand_counts(
                &mut counts_marginal,
                &mut counts_joint,
                partial_clustering.max_label() + 1,
            )
        }
        for i in 0..partial_clustering.n_items_allocated() {
            let item = self.permutation.get(i);
            let label_in_anchor = self.anchor.get(item);
            let label = clustering.allocation()[item];
            let s = self.shrinkage[item];
            counts_marginal[label] += s;
            counts_joint[label_in_anchor][label] += s;
        }
        (partial_clustering, counts_marginal, counts_joint)
    }
}

fn expand_counts(counts_marginal: &mut Vec<f64>, counts_joint: &mut [Vec<f64>], new_len: usize) {
    counts_marginal.resize(new_len, 0.0);
    counts_joint.iter_mut().for_each(|x| x.resize(new_len, 0.0));
}

impl<D: PredictiveProbabilityFunction + Clone> FullConditional for SpParameters<D> {
    // Implement starting only at item and subsequent items.
    fn log_full_conditional(&self, item: usize, clustering: &Clustering) -> Vec<(usize, f64)> {
        let mut target = clustering.allocation().clone();
        let (partial_clustering, counts_marginal, counts_joint) =
            self.prepare_for_partial(item, clustering);
        let candidate_labels = clustering.available_labels_for_reallocation(item);
        candidate_labels
            .map(|label| {
                target[item] = label;
                (
                    label,
                    engine::<D, Pcg64Mcg>(
                        self,
                        partial_clustering.clone(),
                        counts_marginal.clone(),
                        counts_joint.clone(),
                        Some(&target[..]),
                        None,
                    )
                    .1,
                )
            })
            .collect()
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

impl<D: PredictiveProbabilityFunction + Clone> ProbabilityMassFunctionPartial for SpParameters<D> {
    fn log_pmf_partial(&self, item: usize, partition: &Clustering) -> f64 {
        let (partial_clustering, counts_marginal, counts_joint) =
            self.prepare_for_partial(item, partition);
        engine::<D, Pcg64Mcg>(
            self,
            partial_clustering,
            counts_marginal,
            counts_joint,
            Some(&partition.allocation()[..]),
            None,
        )
        .1
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

impl<D: PredictiveProbabilityFunction + Clone> HasGrit for SpParameters<D> {
    fn grit(&self) -> &Grit {
        &self.grit
    }
    fn grit_mut(&mut self) -> &mut Grit {
        &mut self.grit
    }
}

fn engine_full<D: PredictiveProbabilityFunction + Clone, T: Rng>(
    parameters: &SpParameters<D>,
    target: Option<&[usize]>,
    rng: Option<&mut T>,
) -> (Clustering, f64) {
    let m = parameters.anchor.max_label() + 1;
    engine(
        parameters,
        Clustering::unallocated(parameters.anchor.n_items()),
        Vec::new(),
        vec![Vec::new(); m],
        target,
        rng,
    )
}

fn engine<D: PredictiveProbabilityFunction + Clone, T: Rng>(
    parameters: &SpParameters<D>,
    mut clustering: Clustering,
    mut counts_marginal: Vec<f64>,
    mut counts_joint: Vec<Vec<f64>>,
    target: Option<&[usize]>,
    mut rng: Option<&mut T>,
) -> (Clustering, f64) {
    let mut log_probability = 0.0;
    for i in clustering.n_items_allocated()..clustering.n_items() {
        let item = parameters.permutation.get(i);
        let label_in_anchor = parameters.anchor.get(item);
        let shrinkage = parameters.shrinkage[item];
        let shrinkage_scaled = 2.0 * shrinkage / ((i + 1) as f64).powi(2);
        let candidate_labels: Vec<usize> = clustering
            .available_labels_for_allocation_with_target(target, item)
            .collect();
        let max_candidate_label = *candidate_labels.iter().max().unwrap();
        if max_candidate_label >= counts_joint[label_in_anchor].len() {
            expand_counts(
                &mut counts_marginal,
                &mut counts_joint,
                max_candidate_label + 1,
            )
        }
        let labels_and_log_weights = parameters
            .baseline_ppf
            .log_predictive_weight(item, &candidate_labels, &clustering)
            .into_iter()
            .map(|(label, log_probability)| {
                let log_anchor_fidelity = shrinkage_scaled
                    * (counts_joint[label_in_anchor][label].powi(2)
                        - parameters.grit * counts_marginal[label].powi(2));
                let lp = log_probability + log_anchor_fidelity;
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
        counts_marginal[label] += shrinkage;
        counts_joint[label_in_anchor][label] += shrinkage;
    }
    (clustering, log_probability)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crp::CrpParameters;

    #[test]
    fn test_goodness_of_fit_constructive() {
        let n_items = 4;
        let discount = Discount::new(0.1).unwrap();
        let concentration = Concentration::new_with_discount(2.0, discount).unwrap();
        let mut rng = thread_rng();
        for target in Clustering::iter(n_items) {
            let target = Clustering::from_vector(target);
            let mut vec = Vec::with_capacity(target.n_clusters());
            for _ in 0..target.n_items() {
                vec.push(rng.gen_range(0.0..10.0));
            }
            let shrinkage = Shrinkage::from(&vec[..]).unwrap();
            let permutation = Permutation::random(n_items, &mut rng);
            let grit = Grit::one();
            let baseline =
                CrpParameters::new_with_discount(n_items, concentration, discount).unwrap();
            let parameters =
                SpParameters::new(target, shrinkage, permutation, grit, baseline).unwrap();
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
        let discount = Discount::new(0.1).unwrap();
        let concentration = Concentration::new_with_discount(2.0, discount).unwrap();
        let mut rng = thread_rng();
        for target in Clustering::iter(n_items) {
            let target = Clustering::from_vector(target);
            let mut vec = Vec::with_capacity(target.n_clusters());
            for _ in 0..target.n_items() {
                vec.push(rng.gen_range(0.0..10.0));
            }
            let shrinkage = Shrinkage::from(&vec[..]).unwrap();
            let permutation = Permutation::random(n_items, &mut rng);
            let grit = Grit::one();
            let baseline =
                CrpParameters::new_with_discount(n_items, concentration, discount).unwrap();
            let parameters =
                SpParameters::new(target, shrinkage, permutation, grit, baseline).unwrap();
            let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
            crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
        }
    }
}
