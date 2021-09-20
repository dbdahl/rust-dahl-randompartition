// Alternative shrinkage partition distribution

use crate::clust::Clustering;
use crate::distr::{
    FullConditional, PartitionSampler, PredictiveProbabilityFunction, ProbabilityMassFunction,
};
use crate::perm::Permutation;
use crate::shrink::Shrinkage;

use rand::prelude::*;
use rand_pcg::Pcg64Mcg;

#[derive(Debug, Clone)]
pub struct Sp2Parameters<D: PredictiveProbabilityFunction + Clone> {
    baseline_partition: Clustering,
    shrinkage: Shrinkage,
    permutation: Permutation,
    baseline_ppf: D,
}

impl<D: PredictiveProbabilityFunction + Clone> Sp2Parameters<D> {
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

    pub fn shuffle_permutation<T: Rng>(&mut self, rng: &mut T) {
        self.permutation.shuffle(rng);
    }

    pub fn resample_shrinkage<T: Rng>(&mut self, max: f64, shape1: f64, shape2: f64, rng: &mut T) {
        self.shrinkage =
            Shrinkage::constant_random(self.shrinkage.n_items(), max, shape1, shape2, rng);
    }
}

fn expand_counts(counts: &mut Vec<Vec<usize>>, new_len: usize) {
    counts.iter_mut().map(|x| x.resize(new_len, 0)).collect()
}

impl<D: PredictiveProbabilityFunction + Clone> FullConditional for Sp2Parameters<D> {
    // Implement starting only at item and subsequent items.
    fn log_full_conditional(&self, item: usize, clustering: &Clustering) -> Vec<(usize, f64)> {
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

impl<D: PredictiveProbabilityFunction + Clone> PartitionSampler for Sp2Parameters<D> {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering {
        engine_full(self, None, Some(rng)).0
    }
}

impl<D: PredictiveProbabilityFunction + Clone> ProbabilityMassFunction for Sp2Parameters<D> {
    fn log_pmf(&self, partition: &Clustering) -> f64 {
        engine_full::<D, Pcg64Mcg>(self, Some(partition.allocation()), None).1
    }
    fn is_normalized(&self) -> bool {
        true
    }
}

fn engine_full<'a, D: PredictiveProbabilityFunction + Clone, T: Rng>(
    parameters: &'a Sp2Parameters<D>,
    target: Option<&[usize]>,
    rng: Option<&mut T>,
) -> (Clustering, f64) {
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

fn engine<'a, D: PredictiveProbabilityFunction + Clone, T: Rng>(
    parameters: &'a Sp2Parameters<D>,
    mut clustering: Clustering,
    mut counts_marginal: Vec<usize>,
    mut counts: Vec<Vec<usize>>,
    target: Option<&[usize]>,
    mut rng: Option<&mut T>,
) -> (Clustering, f64) {
    // Removing this mapper functionality could speed computations and, therefore, should be removed
    // once the formulation is finalized.
    // In R: Sys.setenv("PUMPKIN_SP2_EXP"="FALSE")
    let mapper = if std::env::var("PUMPKIN_SP2_EXP").unwrap_or_default() == "FALSE" {
        |x: f64| (1.0 + x).ln()
    } else {
        |x: f64| x
    };
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
        let labels_and_log_weights = parameters
            .baseline_ppf
            .log_predictive_weight(item, &candidate_labels, &clustering)
            .into_iter()
            .map(|(label, log_probability)| {
                let n_joint = counts[label_in_baseline][label] as f64;
                (
                    label,
                    log_probability
                        + if n_marginal > 0.0 {
                            mapper(shrinkage * n_joint / n_marginal)
                        } else {
                            0.0
                        },
                )
            });
        let (label, log_probability_contribution) = match &mut rng {
            Some(r) => clustering.select(labels_and_log_weights, true, 0, Some(r), true),
            None => clustering.select::<Pcg64Mcg, _>(
                labels_and_log_weights,
                true,
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
            let baseline_distribution =
                CrpParameters::new_with_mass_and_discount(n_items, mass, discount);
            let parameters =
                Sp2Parameters::new(target, shrinkage, permutation, baseline_distribution).unwrap();
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
                Sp2Parameters::new(target, shrinkage, permutation, baseline_distribution).unwrap();
            let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
            crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
        }
    }
}
