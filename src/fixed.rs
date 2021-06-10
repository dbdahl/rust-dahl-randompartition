// Fixed partition

use crate::clust::Clustering;
use crate::distr::{PartitionSampler, ProbabilityMassFunction};

use rand::Rng;

#[derive(Debug)]
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

    fn is_normalized(&self) -> bool {
        true
    }
}

