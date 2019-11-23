use dahl_partition::*;
use statrs::distribution::{ChiSquared, Univariate};
use std::collections::HashMap;

pub fn assert_goodness_of_fit(
    n_samples: usize,
    n_items: usize,
    mut sample: impl FnMut() -> Partition,
    log_pmf: impl Fn(&mut Partition) -> f64,
    alpha: f64,
) -> () {
    let ns = n_samples as f64;
    let mut map = HashMap::new();
    for _ in 0..n_samples {
        let s = sample();
        let key = s.labels_via_copying();
        *map.entry(key).or_insert(0) += 1;
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
    let distr = ChiSquared::new(df as f64).unwrap();
    let p_value = 1.0 - distr.cdf(chisq);
    assert!(
        p_value > alpha,
        format!(
            "Rejected goodness of fit test.\np-value: {}\nchisq: {}\ndf: {}",
            p_value, chisq, df
        )
    );
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
