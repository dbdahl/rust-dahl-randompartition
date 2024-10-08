use crate::clust::Clustering;
use crate::perm::Permutation;
use crate::prelude::*;
use crate::shrink::Shrinkage;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

//

pub trait PredictiveProbabilityFunction {
    // Clustering is only partially allocated.  Responses are log weights, not necessarily log probabilities.
    fn log_predictive_weight(
        &self,
        item: usize,
        candidate_labels: &[usize],
        clustering: &Clustering,
    ) -> Vec<(usize, f64)>;

    fn predictive_probability(
        &self,
        item: usize,
        candidate_labels: &[usize],
        clustering: &Clustering,
    ) -> Vec<(usize, f64)> {
        let (labels, log_weights): (Vec<_>, Vec<_>) = self
            .log_predictive_weight(item, candidate_labels, clustering)
            .into_iter()
            .unzip();
        let max_log_weight = log_weights.iter().cloned().fold(f64::NAN, f64::max);
        let weights = log_weights
            .iter()
            .map(|x| (*x - max_log_weight).exp())
            .collect::<Vec<_>>();
        let sum: f64 = weights.iter().sum();
        labels
            .into_iter()
            .zip(weights)
            .map(|(label, x)| (label, x / sum))
            .collect()
    }

    fn crp_concentration_ln(&self) -> f64 {
        f64::NEG_INFINITY
    }
}

//

pub trait FullConditional {
    // Clustering is fully allocated, including item.
    fn log_full_conditional(&self, item: usize, clustering: &Clustering) -> Vec<(usize, f64)>;
}

//

pub trait HasPermutation {
    fn permutation(&self) -> &Permutation;
    fn permutation_mut(&mut self) -> &mut Permutation;
}

//

pub trait HasScalarShrinkage {
    fn shrinkage(&self) -> &ScalarShrinkage;
    fn shrinkage_mut(&mut self) -> &mut ScalarShrinkage;
}

//

pub trait HasVectorShrinkage {
    fn shrinkage(&self) -> &Shrinkage;
    fn shrinkage_mut(&mut self) -> &mut Shrinkage;
}

//

pub trait HasGrit {
    fn grit(&self) -> &Grit;
    fn grit_mut(&mut self) -> &mut Grit;
}

//

pub trait PartitionSampler {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering;
}

pub(crate) fn default_partition_sampler_sample<S: Rng, T: PredictiveProbabilityFunction>(
    ppf: &T,
    permutation: &Permutation,
    rng: &mut S,
) -> Clustering {
    let clustering = Clustering::unallocated(permutation.n_items());
    default_partition_conditional_sampler_sample(ppf, permutation, clustering, rng)
}

pub trait PartitionConditionalSampler {
    fn sample_conditionally<T: Rng>(&self, clustering: Clustering, rng: &mut T) -> Clustering;
}

pub(crate) fn default_partition_conditional_sampler_sample<
    S: Rng,
    T: PredictiveProbabilityFunction,
>(
    ppf: &T,
    permutation: &Permutation,
    mut clustering: Clustering,
    rng: &mut S,
) -> Clustering {
    for i in clustering.n_items_allocated()..clustering.n_items() {
        let ii = permutation.get(i);
        let candidate_labels: Vec<_> = clustering.available_labels_for_allocation().collect();
        let labels_and_log_weights = ppf
            .log_predictive_weight(ii, &candidate_labels, &clustering)
            .into_iter();
        let (label, _) =
            clustering.select(labels_and_log_weights, true, false, 0, Some(rng), false);
        clustering.allocate(ii, label);
    }
    clustering
}

//

pub trait NormalizedProbabilityMassFunction {}

pub trait ProbabilityMassFunction {
    fn log_pmf(&self, clustering: &Clustering) -> f64;
}

pub trait ProbabilityMassFunctionPartial {
    fn log_pmf_partial(&self, item: usize, clustering: &Clustering) -> f64;
}

pub(crate) fn default_probability_mass_function_log_pmf<T: PredictiveProbabilityFunction>(
    ppf: &T,
    permutation: &Permutation,
    clustering: &Clustering,
) -> f64 {
    let target = &clustering.allocation()[..];
    let n_items = permutation.n_items();
    let mut working_clustering = Clustering::unallocated(n_items);
    let mut log_prob = 0.0;
    for i in 0..working_clustering.n_items() {
        let ii = permutation.get(i);
        let labels_and_log_weights = ppf
            .log_predictive_weight(
                ii,
                &working_clustering
                    .available_labels_for_allocation_with_target(Some(target), ii)
                    .collect::<Vec<_>>()[..],
                &working_clustering,
            )
            .into_iter();
        let (label, contribution) = working_clustering.select::<Pcg64Mcg, _>(
            labels_and_log_weights,
            true,
            false,
            target[ii],
            None,
            true,
        );
        log_prob += contribution;
        working_clustering.allocate(ii, label);
    }
    log_prob
}
