use crate::clust::Clustering;
use crate::perm::Permutation;
use rand::Rng;

//

pub trait PredictiveProbabilityFunctionOld {
    // Item may already be allocated somewhere in clustering.
    fn log_predictive_probability(&self, item: usize, label: usize, clustering: &Clustering)
        -> f64;
}

//

pub trait PredictiveProbabilityFunction {
    // Clustering is only partially allocated.
    fn log_predictive(
        &self,
        item: usize,
        candidate_labels: Vec<usize>,
        clustering: &Clustering,
    ) -> Vec<(usize, f64)>;
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
    clustering.allocate(0, 0);
    for i in 1..clustering.n_items() {
        let ii = permutation.get(i);
        let labels_and_log_weights = ppf
            .log_predictive(
                ii,
                clustering.available_labels_for_allocation().collect(),
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

pub(crate) fn default_probability_mass_function_log_pmf_without_permutation() -> f64 {
    unimplemented!()
}
