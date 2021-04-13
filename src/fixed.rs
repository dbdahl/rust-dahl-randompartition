// Fixed partition

use crate::clust::Clustering;
use crate::distr::{PartitionSampler, ProbabilityMassFunction};

use rand::Rng;
use std::slice;

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

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__fixedpartitionparameters_new(
    n_items: i32,
    clustering_ptr: *const i32,
) -> *mut FixedPartitionParameters {
    let ni = n_items as usize;
    let clustering = Clustering::from_slice(slice::from_raw_parts(clustering_ptr, ni));
    // First we create a new object.
    let obj = FixedPartitionParameters::new(clustering);
    // Then copy it to the heap (so we have a stable pointer to it).
    let boxed_obj = Box::new(obj);
    // Then return a pointer by converting our `Box<_>` into a raw pointer
    Box::into_raw(boxed_obj)
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__fixedpartitionparameters_free(
    obj: *mut FixedPartitionParameters,
) {
    // As a rule of thumb, freeing a null pointer is just a noop.
    if obj.is_null() {
        return;
    }
    // Convert the raw pointer back to a Box<_>
    let boxed = Box::from_raw(obj);
    // Then explicitly drop it (optional)
    drop(boxed);
}
