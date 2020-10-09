// Centered partition process

use crate::crp::{log_pmf as crp_log_pmf, CRPParameters};
use crate::mcmc::NealFunctionsGeneral;
use crate::prelude::*;

use dahl_partition::*;
use dahl_salso::clustering::{Clusterings, WorkingClustering};
use dahl_salso::log2cache::Log2Cache;
use dahl_salso::optimize::{BinderCMLossComputer, CMLossComputer, VICMLossComputer};
use std::slice;

pub struct CPPParameters<'a> {
    center: &'a Partition,
    rate: Rate,
    mass: Mass,
    discount: Discount,
    use_vi: bool,
    a: f64,
    log2cache: Log2Cache,
    center_as_clusterings: Clusterings,
}

impl<'a> CPPParameters<'a> {
    pub fn use_vi(
        center: &'a Partition,
        rate: Rate,
        mass: Mass,
        discount: Discount,
    ) -> Option<Self> {
        Self::new(center, rate, mass, discount, true, 1.0)
    }

    pub fn use_binder(
        center: &'a Partition,
        rate: Rate,
        mass: Mass,
        discount: Discount,
    ) -> Option<Self> {
        Self::new(center, rate, mass, discount, false, 1.0)
    }

    pub fn new(
        center: &'a Partition,
        rate: Rate,
        mass: Mass,
        discount: Discount,
        use_vi: bool,
        a: f64,
    ) -> Option<Self> {
        let labels: Vec<_> = center.labels().iter().map(|x| x.unwrap() as i32).collect();
        let center_as_clusterings =
            Clusterings::from_i32_column_major_order(&labels[..], center.n_items());
        Some(Self {
            center,
            rate,
            mass,
            discount,
            use_vi,
            a,
            log2cache: Log2Cache::new(center.n_items()),
            center_as_clusterings,
        })
    }
}

impl<'a> NealFunctionsGeneral for CPPParameters<'a> {
    fn log_weight(&self, item_index: usize, subset_index: usize, partition: &Partition) -> f64 {
        let mut p = partition.clone();
        p.add_with_index(item_index, subset_index);
        log_pmf(&p, self)
    }
}

pub fn log_pmf(target: &Partition, parameters: &CPPParameters) -> f64 {
    let computer: Box<dyn CMLossComputer> = if parameters.use_vi {
        Box::new(VICMLossComputer::new(parameters.a, &parameters.log2cache))
    } else {
        Box::new(BinderCMLossComputer::new(parameters.a))
    };
    let target_as_vector: Vec<_> = target
        .labels()
        .iter()
        .map(|x| x.unwrap() as dahl_salso::LabelType)
        .collect();
    let target_as_working = WorkingClustering::from_vector(
        target_as_vector,
        target.n_subsets() as dahl_salso::LabelType,
    );
    let distance = computer.compute_loss(
        &target_as_working,
        &parameters
            .center_as_clusterings
            .make_confusion_matrices(&target_as_working),
    );
    let crp_parameters =
        CRPParameters::new_with_mass_and_discount(parameters.mass, parameters.discount);
    crp_log_pmf(target, &crp_parameters) - parameters.rate.unwrap() * distance
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_pmf() {
        let n_items = 5;
        let discount = 0.1;
        let mass = Mass::new_with_variable_constraint(2.0, discount);
        let discount = Discount::new(discount);
        let mut rng = thread_rng();
        for center in Partition::iter(n_items) {
            let center = Partition::from(&center[..]);
            let rate = Rate::new(rng.gen_range(0.0, 10.0));
            let parameters = CPPParameters::use_vi(&center, rate, mass, discount).unwrap();
            let log_prob_closure = |partition: &mut Partition| log_pmf(partition, &parameters);
            crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
        }
    }
}

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
    let center = Partition::from(slice::from_raw_parts(center_ptr, ni));
    let rate = Rate::new(rate);
    let mass = Mass::new_with_variable_constraint(mass, discount);
    let discount = Discount::new(discount);
    let matrix: &mut [i32] = slice::from_raw_parts_mut(partition_labels_ptr, np * ni);
    let probs: &mut [f64] = slice::from_raw_parts_mut(partition_probs_ptr, np);
    let parameters = CPPParameters::new(&center, rate, mass, discount, use_vi != 0, a).unwrap();
    for i in 0..np {
        let mut target_labels = Vec::with_capacity(ni);
        for j in 0..ni {
            target_labels.push(matrix[np * j + i]);
        }
        let target = Partition::from(&target_labels[..]);
        probs[i] = log_pmf(&target, &parameters);
    }
}
