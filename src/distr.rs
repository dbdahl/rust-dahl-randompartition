use crate::clust::Clustering;
use crate::perm::Permutation;

use rand::Rng;
use rand_isaac::IsaacRng;

//

pub trait PredictiveProbabilityFunction {
    // Clustering is only partially allocated.  Responses are log weights, not necessarily log probabilities.
    fn log_predictive_weight(
        &self,
        item: usize,
        candidate_labels: &Vec<usize>,
        clustering: &Clustering,
    ) -> Vec<(usize, f64)>;

    fn predictive_probability(
        &self,
        item: usize,
        candidate_labels: &Vec<usize>,
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
            .zip(weights.into_iter())
            .map(|(label, x)| (label, x / sum))
            .collect()
    }
}

//

pub trait FullConditional {
    // Clustering is fully allocated, including item.
    fn log_full_conditional(&self, item: usize, clustering: &Clustering) -> Vec<(usize, f64)>;
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
    let n_items = permutation.n_items();
    let mut clustering = Clustering::unallocated(n_items);
    for i in 0..clustering.n_items() {
        let ii = permutation.get(i);
        let labels_and_log_weights = ppf
            .log_predictive_weight(
                ii,
                &clustering.available_labels_for_allocation().collect(),
                &clustering,
            )
            .into_iter();
        let (label, _) = clustering.select(labels_and_log_weights, true, 0, Some(rng), false);
        clustering.allocate(ii, label);
    }
    clustering
}

//

pub trait ProbabilityMassFunction {
    fn log_pmf(&self, clustering: &Clustering) -> f64;
    fn is_normalized(&self) -> bool;
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
                    .collect(),
                &working_clustering,
            )
            .into_iter();
        let (label, contribution) = working_clustering.select::<IsaacRng, _>(
            labels_and_log_weights,
            true,
            target[ii],
            None,
            true,
        );
        log_prob += contribution;
        working_clustering.allocate(ii, label);
    }
    log_prob
}
