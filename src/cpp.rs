// Centered partition process

use crate::clust::Clustering;
use crate::crp::CRPParameters;
use crate::mcmc::PriorLogWeight;
use crate::prelude::*;

use crate::prior::PartitionLogProbability;
use dahl_salso::clustering::Clusterings;
use dahl_salso::clustering::WorkingClustering;
use dahl_salso::log2cache::Log2Cache;
use dahl_salso::optimize::{BinderCMLossComputer, CMLossComputer, VICMLossComputer};
use std::slice;

pub struct CPPParameters {
    center: Clustering,
    rate: Rate,
    mass: Mass,
    discount: Discount,
    use_vi: bool,
    a: f64,
    log2cache: Log2Cache,
    center_as_clusterings: Clusterings,
}

impl CPPParameters {
    pub fn use_vi(center: Clustering, rate: Rate, mass: Mass, discount: Discount) -> Option<Self> {
        Self::new(center, rate, mass, discount, true, 1.0)
    }

    pub fn use_binder(
        center: Clustering,
        rate: Rate,
        mass: Mass,
        discount: Discount,
    ) -> Option<Self> {
        Self::new(center, rate, mass, discount, false, 1.0)
    }

    pub fn new(
        center: Clustering,
        rate: Rate,
        mass: Mass,
        discount: Discount,
        use_vi: bool,
        a: f64,
    ) -> Option<Self> {
        let labels: Vec<_> = center.allocation().iter().map(|x| *x as i32).collect();
        let n_items = center.n_items();
        let center_as_clusterings = Clusterings::from_i32_column_major_order(&labels[..], n_items);
        Some(Self {
            center,
            rate,
            mass,
            discount,
            use_vi,
            a,
            log2cache: Log2Cache::new(n_items),
            center_as_clusterings,
        })
    }
}

impl PriorLogWeight for CPPParameters {
    fn log_weight(&self, item_index: usize, subset_index: usize, clustering: &Clustering) -> f64 {
        let mut p = clustering.clone();
        p.reallocate(item_index, subset_index);
        log_pmf(&p, self)
    }
}

impl PartitionLogProbability for CPPParameters {
    fn log_probability(&self, partition: &Clustering) -> f64 {
        log_pmf(partition, self)
    }
    fn is_normalized(&self) -> bool {
        false
    }
}

fn log_pmf(target: &Clustering, parameters: &CPPParameters) -> f64 {
    let computer: Box<dyn CMLossComputer> = if parameters.use_vi {
        Box::new(VICMLossComputer::new(parameters.a, &parameters.log2cache))
    } else {
        Box::new(BinderCMLossComputer::new(parameters.a))
    };
    let target_as_vector: Vec<_> = target
        .allocation()
        .iter()
        .map(|x| *x as dahl_salso::LabelType)
        .collect();
    let target_as_working = WorkingClustering::from_vector(
        target_as_vector,
        (target.max_label() + 1) as dahl_salso::LabelType,
    );
    let distance = computer.compute_loss(
        &target_as_working,
        &parameters
            .center_as_clusterings
            .make_confusion_matrices(&target_as_working),
    );
    let crp_parameters = CRPParameters::new_with_mass_and_discount(
        parameters.mass,
        parameters.discount,
        parameters.center.n_items(),
    );
    crp_parameters.log_probability(target) - parameters.rate.unwrap() * distance
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    #[should_panic]
    fn test_pmf() {
        let n_items = 5;
        let discount = 0.1;
        let mass = Mass::new_with_variable_constraint(2.0, discount);
        let discount = Discount::new(discount);
        let mut rng = thread_rng();
        for center in Clustering::iter(n_items) {
            let center = Clustering::from_vector(center);
            let rate = Rate::new(rng.gen_range(0.0, 10.0));
            //let rate = Rate::new(0.0);
            let parameters = CPPParameters::use_vi(center, rate, mass, discount).unwrap();
            let log_prob_closure =
                |clustering: &mut Clustering| parameters.log_probability(clustering);
            // Their method does NOT sum to one!  Hence "#[should_panic]" above.
            crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__cppparameters_new(
    n_items: i32,
    focal_ptr: *const i32,
    rate: f64,
    mass: f64,
    discount: f64,
    use_vi: bool,
    a: f64,
) -> *mut CPPParameters {
    let ni = n_items as usize;
    let focal = Clustering::from_slice(slice::from_raw_parts(focal_ptr, ni));
    let r = Rate::new(rate);
    let d = Discount::new(discount);
    let m = Mass::new_with_variable_constraint(mass, discount);
    // First we create a new object.
    let obj = CPPParameters::new(focal, r, m, d, use_vi, a).unwrap();
    // Then copy it to the heap (so we have a stable pointer to it).
    let boxed_obj = Box::new(obj);
    // Then return a pointer by converting our `Box<_>` into a raw pointer
    Box::into_raw(boxed_obj)
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__cppparameters_free(obj: *mut CPPParameters) {
    // As a rule of thumb, freeing a null pointer is just a noop.
    if obj.is_null() {
        return;
    }
    // Convert the raw pointer back to a Box<_>
    let boxed = Box::from_raw(obj);
    // Then explicitly drop it (optional)
    drop(boxed);
}

/*
#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__centered_partition(
    n_partitions: i32,
    n_items: i32,
    partition_labels_ptr: *mut i32,
    partition_probs_ptr: *mut f64,
    center_ptr: *const i32,
    rate: f64,
    mass: f64,
    discount: f64,
    use_vi: i32,
    a: f64,
) -> () {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let center = Clustering::from_slice(slice::from_raw_parts(center_ptr, ni));
    let rate = Rate::new(rate);
    let mass = Mass::new_with_variable_constraint(mass, discount);
    let discount = Discount::new(discount);
    let matrix: &mut [i32] = slice::from_raw_parts_mut(partition_labels_ptr, np * ni);
    let probs: &mut [f64] = slice::from_raw_parts_mut(partition_probs_ptr, np);
    let parameters = CPPParameters::new(center, rate, mass, discount, use_vi != 0, a).unwrap();
    for i in 0..np {
        let mut target_labels = Vec::with_capacity(ni);
        for j in 0..ni {
            target_labels.push(matrix[np * j + i] as usize);
        }
        let target = Clustering::from_vector(target_labels);
        probs[i] = log_pmf(&target, &parameters);
    }
}
*/
