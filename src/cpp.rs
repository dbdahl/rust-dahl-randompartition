// Centered partition process

use crate::clust::Clustering;
use crate::distr::{FullConditional, ProbabilityMassFunction};

#[derive(Debug, Clone)]
pub struct CppParameters<D: ProbabilityMassFunction> {
    pub anchor: Clustering,
    rate: f64,
    baseline_pmf: D,
    use_vi: bool,
    a: f64,
    count_terms: Vec<f64>,
}

impl<D: ProbabilityMassFunction> CppParameters<D> {
    pub fn new(
        anchor: Clustering,
        rate: f64,
        baseline_pmf: D,
        use_vi: bool,
        a: f64,
    ) -> Option<Self> {
        if anchor.n_items() == 0 || !rate.is_finite() || rate < 0.0 || !a.is_finite() || a < 0.0 {
            return None;
        }
        // Binder uses squared counts; variation of information uses n log2(n).
        let mut count_terms = Vec::with_capacity(anchor.n_items() + 1);
        count_terms.push(0.0);
        count_terms.extend((1..=anchor.n_items()).map(|n| {
            let n = n as f64;
            if use_vi { n * n.log2() } else { n * n }
        }));
        Some(Self {
            anchor,
            rate,
            baseline_pmf,
            use_vi,
            a,
            count_terms,
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

// Group the target labels by anchor cluster, then reuse one row of counts.
// The last bin in each label space is for a partially allocated clustering.
fn joint_sum(anchor: &Clustering, target: &Clustering, terms: &[f64]) -> f64 {
    let mut ends = Vec::with_capacity(anchor.sizes().len() + 1);
    let mut offset = 0;
    for &size in anchor.sizes() {
        ends.push(offset);
        offset += size;
    }
    ends.push(offset);
    let mut grouped = vec![0; anchor.n_items()];
    for (&a, &b) in anchor.allocation().iter().zip(target.allocation()) {
        let end = &mut ends[a.min(anchor.sizes().len())];
        grouped[*end] = b.min(target.sizes().len());
        *end += 1;
    }
    let mut counts = vec![0usize; target.sizes().len() + 1];
    let mut joint = 0.0;
    let mut start = 0;
    for end in ends {
        let labels = &grouped[start..end];
        for &label in labels {
            counts[label] += 1;
        }
        for &label in labels {
            // terms[0] is zero, so each occupied cell contributes once.
            joint += terms[std::mem::take(&mut counts[label])];
        }
        start = end;
    }
    joint
}

fn log_pmf<D: ProbabilityMassFunction>(target: &Clustering, parameters: &CppParameters<D>) -> f64 {
    let anchor = &parameters.anchor;
    let n = anchor.n_items();
    assert_eq!(target.n_items(), n);
    let terms = &parameters.count_terms;
    let marginal =
        |p: &Clustering| -> f64 { p.active_labels().iter().map(|&k| terms[p.size_of(k)]).sum() };
    let joint = joint_sum(anchor, target, terms);
    let a = parameters.a;
    let distance = if parameters.use_vi {
        ((2.0 - a) * marginal(target) + (a * marginal(anchor) - 2.0 * joint)) / n as f64
    } else {
        (a * marginal(anchor) + (2.0 - a) * marginal(target) - 2.0 * joint) / terms[n]
    };
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
    fn loss_uses_partitions_independently_of_label_width() {
        let n = 65_536;
        let anchor = Clustering::from_vector((0..n).map(|i| i % 2).collect());
        let sparse = Clustering::from_vector((0..n).map(|i| (i % 2) * n).collect());
        for vi in [false, true] {
            let baseline = CrpParameters::new(n, Concentration::new(1.0).unwrap());
            let p = CppParameters::new(anchor.clone(), 2.0, baseline, vi, 1.0).unwrap();
            assert_eq!(p.log_pmf(&anchor), p.log_pmf(&sparse));

            let singletons = Clustering::singleton_clusters(n);
            let baseline = CrpParameters::new(n, Concentration::new(1.0).unwrap());
            let p = CppParameters::new(singletons.clone(), 2.0, baseline.clone(), vi, 1.0).unwrap();
            assert_eq!(p.log_pmf(&singletons), baseline.log_pmf(&singletons));
        }
    }

    #[test]
    #[should_panic]
    fn test_pmf() {
        let n_items = 5;
        let discount = Discount::new(0.1).unwrap();
        let concentration = Concentration::new_with_discount(2.0, discount).unwrap();
        let mut rng = rand::rng();
        for anchor in Clustering::iter(n_items) {
            let anchor = Clustering::from_vector(anchor);
            let rate = loop {
                let x = rng.random_range(0.0..10.0);
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
