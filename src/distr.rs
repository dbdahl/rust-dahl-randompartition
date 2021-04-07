use crate::clust::Clustering;
use crate::perm::Permutation;
use rand::Rng;

//

pub trait PredictiveProbabilityFunction {
    // Item may already be allocated somewhere in clustering.
    fn log_predictive_probability(&self, item: usize, label: usize, clustering: &Clustering)
        -> f64;
}

//

pub trait FullConditional {
    // Clustering if fully allocated.
    fn log_full_conditional(&self, item: usize, clustering: &Clustering) -> Vec<(usize, f64)>;
}

pub(crate) fn full_conditional_log_full_conditional_exchangeable_default<T>(
    ppf: &T,
    item: usize,
    clustering: &Clustering,
) -> Vec<(usize, f64)>
where
    T: PredictiveProbabilityFunction,
{
    clustering
        .available_labels_for_reallocation(item)
        .map(|label| {
            (
                label,
                ppf.log_predictive_probability(item, label, clustering),
            )
        })
        .collect()
}

//

pub trait PartitionSampler {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering;
}

pub(crate) fn default_partition_sampler_sample_without_permutation<
    S: Rng,
    T: PredictiveProbabilityFunction,
>(
    ppf: &T,
    n_items: usize,
    rng: &mut S,
) -> Clustering {
    let mut clustering = Clustering::unallocated(n_items);
    clustering.allocate(0, 0);
    for i in 1..clustering.n_items() {
        let labels_and_log_weights = clustering.available_labels_for_allocation().map(|label| {
            let log_weight = ppf.log_predictive_probability(i, label, &clustering);
            (label, log_weight)
        });
        let (label, _) = clustering.select(labels_and_log_weights, true, 0, Some(rng), false);
        clustering.allocate(i, label);
    }
    clustering
}

pub(crate) fn default_partition_sampler_sample_with_permutation<
    S: Rng,
    T: PredictiveProbabilityFunction,
>(
    ppf: &T,
    permutation: &Permutation,
    rng: &mut S,
) -> Clustering {
    let n_items = permutation.n_items();
    let mut clustering = Clustering::unallocated(n_items);
    clustering.allocate(permutation.get(0), 0);
    for i in 1..n_items {
        let ii = permutation.get(i);
        let labels_and_log_weights = clustering.available_labels_for_allocation().map(|label| {
            let log_weight = ppf.log_predictive_probability(ii, label, &clustering);
            (label, log_weight)
        });
        let (label, _) = clustering.select(labels_and_log_weights, true, 0, Some(rng), false);
        clustering.allocate(i, label);
    }
    clustering
}

//

pub trait ProbabilityMassFunction {
    fn log_pmf(&self, clustering: &Clustering) -> f64;
    fn is_normalized(&self) -> bool;
}

pub(crate) fn default_probability_mass_function_log_pmf_without_permutation() -> f64 {
    unimplemented!()
}
