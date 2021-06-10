// Centered partition process

use crate::clust::Clustering;
use crate::crp::CrpParameters;
use crate::distr::{FullConditional, ProbabilityMassFunction};
use crate::prelude::*;

use dahl_salso::clustering::Clusterings;
use dahl_salso::clustering::WorkingClustering;
use dahl_salso::log2cache::Log2Cache;
use dahl_salso::optimize::{BinderCMLossComputer, CMLossComputer, VICMLossComputer};

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

impl FullConditional for CppParameters {
    fn log_full_conditional<'a>(
        &'a self,
        item: usize,
        clustering: &'a Clustering,
    ) -> Vec<(usize, f64)> {
        clustering
            .available_labels_for_reallocation(item)
            .map(|label| {
                let mut p = clustering.clone();
                p.allocate(item, label);
                (label, log_pmf(&p, self))
            })
            .collect()
    }
}

impl ProbabilityMassFunction for CppParameters {
    fn log_pmf(&self, partition: &Clustering) -> f64 {
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
            parameters.baseline_partition.n_items(),
            parameters.mass,
            parameters.discount,
        );
        crp_parameters.log_pmf(target) + log_multiplier
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
            let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
            // Their method does NOT sum to one!  Hence "#[should_panic]" above.
            crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
        }
    }
}
