use dahl_partition::*;
use statrs::distribution::{ChiSquared, Univariate};
use std::collections::HashMap;

pub fn assert_goodness_of_fit(
    n_samples: usize,
    n_items: usize,
    mut sample: impl FnMut() -> Partition,
    log_pmf: impl Fn(&mut Partition) -> f64,
    n_calls_per_sample: usize,
    alpha: f64,
) -> Option<String> {
    let ns = n_samples as f64;
    let mut map = HashMap::new();
    for i in 0..(n_calls_per_sample * n_samples) {
        let s = sample();
        if (i + 1) % n_calls_per_sample == 0 {
            let key = s.labels_via_copying();
            *map.entry(key).or_insert(0) += 1;
        }
    }
    let threshold = 5.0;
    let mut chisq = 0.0;
    let mut df = 0;
    let mut observed = 0;
    let mut expected = 0.0;
    for labels in Partition::iter(n_items) {
        let slice = &labels[..];
        observed += *map.get(slice).unwrap_or(&0);
        expected += ns * log_pmf(&mut Partition::from(slice)).exp();
        if expected >= threshold {
            let o = observed as f64;
            chisq += (o - expected) * (o - expected) / expected;
            df += 1;
            observed = 0;
            expected = 0.0;
        }
    }
    let distr = ChiSquared::new((df - 1) as f64).unwrap();
    let p_value = 1.0 - distr.cdf(chisq);
    if p_value <= alpha {
        Some(format!(
            "Rejected goodness of fit test... p-value: {:.8}, chisq: {:.2}, df: {}",
            p_value, chisq, df
        ))
    } else {
        None
    }
}

pub fn assert_pmf_sums_to_one(
    n_items: usize,
    log_pmf: impl Fn(&mut Partition) -> f64,
    epsilon: f64,
) -> () {
    let sum = Partition::iter(n_items)
        .map(|p| log_pmf(&mut Partition::from(&p[..])).exp())
        .sum();
    assert!(
        1.0 - epsilon <= sum && sum <= 1.0 + epsilon,
        format!("Total probability should be one, but is {}.", sum)
    );
}
