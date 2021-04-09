// Centered partition process

use crate::clust::Clustering;
use crate::crp::CrpParameters;
use crate::distr::PredictiveProbabilityFunctionOld;
use crate::prelude::*;

use crate::prior::PartitionLogProbability;
use dahl_salso::clustering::Clusterings;
use dahl_salso::clustering::WorkingClustering;
use dahl_salso::log2cache::Log2Cache;
use dahl_salso::optimize::{BinderCMLossComputer, CMLossComputer, VICMLossComputer};
use std::slice;

#[derive(Debug, Clone)]
pub struct CppParameters {
    baseline_partition: Clustering,
    rate: Rate,
    uniform: bool,
    mass: Mass,
    discount: Discount,
    use_vi: bool,
    a: f64,
    log2cache: Log2Cache,
    baseline_as_clusterings: Clusterings,
}

impl CppParameters {
    pub fn use_vi(
        baseline: Clustering,
        rate: Rate,
        uniform: bool,
        mass: Mass,
        discount: Discount,
    ) -> Option<Self> {
        Self::new(baseline, rate, uniform, mass, discount, true, 1.0)
    }

    pub fn use_binder(
        baseline: Clustering,
        rate: Rate,
        uniform: bool,
        mass: Mass,
        discount: Discount,
    ) -> Option<Self> {
        Self::new(baseline, rate, uniform, mass, discount, false, 1.0)
    }

    pub fn new(
        baseline: Clustering,
        rate: Rate,
        uniform: bool,
        mass: Mass,
        discount: Discount,
        use_vi: bool,
        a: f64,
    ) -> Option<Self> {
        let labels: Vec<_> = baseline.allocation().iter().map(|x| *x as i32).collect();
        let n_items = baseline.n_items();
        let baseline_as_clusterings =
            Clusterings::from_i32_column_major_order(&labels[..], n_items);
        Some(Self {
            baseline_partition: baseline,
            rate,
            uniform,
            mass,
            discount,
            use_vi,
            a,
            log2cache: Log2Cache::new(n_items),
            baseline_as_clusterings,
        })
    }
}

impl PredictiveProbabilityFunctionOld for CppParameters {
    fn log_predictive_probability(
        &self,
        item_index: usize,
        subset_index: usize,
        clustering: &Clustering,
    ) -> f64 {
        let mut p = clustering.clone();
        p.allocate(item_index, subset_index);
        log_pmf(&p, self)
    }
}

impl PartitionLogProbability for CppParameters {
    fn log_probability(&self, partition: &Clustering) -> f64 {
        log_pmf(partition, self)
    }
    fn is_normalized(&self) -> bool {
        false
    }
}

fn log_pmf(target: &Clustering, parameters: &CppParameters) -> f64 {
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
            .baseline_as_clusterings
            .make_confusion_matrices(&target_as_working),
    );
    let log_multiplier = -parameters.rate.unwrap() * distance;
    if parameters.uniform {
        log_multiplier
    } else {
        let crp_parameters = CrpParameters::new_with_mass_and_discount(
            parameters.mass,
            parameters.discount,
            parameters.baseline_partition.n_items(),
        );
        crp_parameters.log_probability(target) + log_multiplier
    }
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
        for baseline in Clustering::iter(n_items) {
            let baseline = Clustering::from_vector(baseline);
            let rate = Rate::new(rng.gen_range(0.0..10.0));
            //let rate = Rate::new(0.0);
            let parameters = CppParameters::use_vi(baseline, rate, false, mass, discount).unwrap();
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
    baseline_ptr: *const i32,
    rate: f64,
    uniform: bool,
    mass: f64,
    discount: f64,
    use_vi: bool,
    a: f64,
) -> *mut CppParameters {
    let ni = n_items as usize;
    let baseline = Clustering::from_slice(slice::from_raw_parts(baseline_ptr, ni));
    let r = Rate::new(rate);
    let d = Discount::new(discount);
    let m = Mass::new_with_variable_constraint(mass, discount);
    // First we create a new object.
    let obj = CppParameters::new(baseline, r, uniform, m, d, use_vi, a).unwrap();
    // Then copy it to the heap (so we have a stable pointer to it).
    let boxed_obj = Box::new(obj);
    // Then return a pointer by converting our `Box<_>` into a raw pointer
    Box::into_raw(boxed_obj)
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__cppparameters_free(obj: *mut CppParameters) {
    // As a rule of thumb, freeing a null pointer is just a noop.
    if obj.is_null() {
        return;
    }
    // Convert the raw pointer back to a Box<_>
    let boxed = Box::from_raw(obj);
    // Then explicitly drop it (optional)
    drop(boxed);
}
