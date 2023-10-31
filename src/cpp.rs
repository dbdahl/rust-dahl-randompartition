// Centered partition process

use crate::clust::Clustering;
use crate::distr::{FullConditional, ProbabilityMassFunction};

use dahl_salso::clustering::Clusterings;
use dahl_salso::clustering::WorkingClustering;
use dahl_salso::log2cache::Log2Cache;
use dahl_salso::optimize::{BinderCMLossComputer, CMLossComputer, VICMLossComputer};

#[derive(Debug, Clone)]
pub struct CppParameters<D: ProbabilityMassFunction> {
    anchor: Clustering,
    rate: f64,
    baseline_pmf: D,
    use_vi: bool,
    a: f64,
    log2cache: Log2Cache,
    baseline_as_clusterings: Clusterings,
}

impl<D: ProbabilityMassFunction> CppParameters<D> {
    pub fn new(
        anchor: Clustering,
        rate: f64,
        baseline_pmf: D,
        use_vi: bool,
        a: f64,
    ) -> Option<Self> {
        if rate.is_nan() || rate.is_infinite() || rate < 0.0 {
            return None;
        }
        let labels: Vec<_> = anchor.allocation().iter().map(|x| *x as i32).collect();
        let n_items = anchor.n_items();
        let baseline_as_clusterings =
            Clusterings::from_i32_column_major_order(&labels[..], n_items);
        Some(Self {
            anchor,
            rate,
            baseline_pmf,
            use_vi,
            a,
            log2cache: Log2Cache::new(n_items),
            baseline_as_clusterings,
        })
    }
}

impl<D: ProbabilityMassFunction> FullConditional for CppParameters<D> {
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

impl<D: ProbabilityMassFunction> ProbabilityMassFunction for CppParameters<D> {
    fn log_pmf(&self, partition: &Clustering) -> f64 {
        log_pmf(partition, self)
    }
}

fn log_pmf<D: ProbabilityMassFunction>(target: &Clustering, parameters: &CppParameters<D>) -> f64 {
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
    let log_multiplier = -parameters.rate * distance;
    let log_baseline_pmf = parameters.baseline_pmf.log_pmf(target);
    log_baseline_pmf + log_multiplier
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crp::CrpParameters;
    use crate::prelude::*;
    use rand::prelude::*;

    #[test]
    #[should_panic]
    fn test_pmf() {
        let n_items = 5;
        let discount = Discount::new(0.1).unwrap();
        let concentration = Concentration::new_with_discount(2.0, discount).unwrap();
        let mut rng = thread_rng();
        for anchor in Clustering::iter(n_items) {
            let anchor = Clustering::from_vector(anchor);
            let rate = loop {
                let x = rng.gen_range(0.0..10.0);
                if x > 0.0 {
                    break x;
                }
            };
            let baseline =
                CrpParameters::new_with_discount(n_items, concentration, discount).unwrap();
            let parameters = CppParameters::new(anchor, rate, baseline, true, 1.0).unwrap();
            let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
            // Their method does NOT sum to one!  Hence "#[should_panic]" above.
            crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
        }
    }
}
