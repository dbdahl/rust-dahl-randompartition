// Fixed partition

use crate::clust::Clustering;
use crate::distr::{
    FullConditional, NormalizedProbabilityMassFunction, PartitionSampler, ProbabilityMassFunction,
};

use rand::Rng;

#[derive(Debug, Clone)]
pub struct FixedPartitionParameters {
    clustering: Clustering,
}

impl FixedPartitionParameters {
    pub fn new(clustering: Clustering) -> Self {
        Self {
            clustering: clustering.standardize(),
        }
    }
}

impl PartitionSampler for FixedPartitionParameters {
    fn sample<T: Rng>(&self, _rng: &mut T) -> Clustering {
        self.clustering.clone()
    }
}

impl ProbabilityMassFunction for FixedPartitionParameters {
    fn log_pmf(&self, partition: &Clustering) -> f64 {
        if partition.standardize().allocation() == self.clustering.allocation() {
            0.0
        } else {
            f64::NEG_INFINITY
        }
    }
}

impl NormalizedProbabilityMassFunction for FixedPartitionParameters {}

impl FullConditional for FixedPartitionParameters {
    fn log_full_conditional(&self, item: usize, clustering: &Clustering) -> Vec<(usize, f64)> {
        let current_label = clustering.get(item);
        clustering
            .available_labels_for_reallocation(item)
            .map(|label| {
                let value = if label == current_label {
                    0.0
                } else {
                    f64::NEG_INFINITY
                };
                (label, value)
            })
            .collect()
    }
}
