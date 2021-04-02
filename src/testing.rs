use crate::clust::Clustering;
use statrs::distribution::{ChiSquared, Univariate};
use std::collections::HashMap;

pub fn assert_goodness_of_fit(
    n_samples: usize,
    n_items: usize,
    mut sample: impl FnMut() -> Clustering,
    log_pmf: impl Fn(&mut Clustering) -> f64,
    n_calls_per_sample: usize,
    alpha: f64,
) {
    let ns = n_samples as f64;
    let mut map = HashMap::new();
    for i in 0..(n_calls_per_sample * n_samples) {
        let s = sample().standardize();
        if (i + 1) % n_calls_per_sample == 0 {
            let key = s.allocation().clone();
            *map.entry(key).or_insert(0) += 1;
        }
    }
    let threshold = 5.0;
    let mut chisq = 0.0;
    let mut df = 0;
    let mut observed = 0;
    let mut expected = 0.0;
    for labels in Clustering::iter(n_items) {
        observed += *map.get(&labels).unwrap_or(&0);
        expected += ns * log_pmf(&mut Clustering::from_vector(labels)).exp();
        if expected >= threshold {
            let o = observed as f64;
            chisq += (o - expected) * (o - expected) / expected;
            df += 1;
            observed = 0;
            expected = 0.0;
        }
    }
    df -= 1;
    let p_value = if df > 0 {
        let distr = ChiSquared::new(df as f64).unwrap();
        1.0 - distr.cdf(chisq)
    } else {
        println!("Warning.... insufficient degrees of freedom!");
        1.0
    };
    if p_value <= alpha {
        panic!(
            "Rejected goodness of fit test... p-value: {:.8}, chisq: {:.2}, df: {}",
            p_value, chisq, df
        );
    }
}

pub fn assert_pmf_sums_to_one(
    n_items: usize,
    log_pmf: impl Fn(&mut Clustering) -> f64,
    epsilon: f64,
) {
    let sum = Clustering::iter(n_items)
        .map(|p| log_pmf(&mut Clustering::from_vector(p)).exp())
        .sum();
    assert!(
        1.0 - epsilon <= sum && sum <= 1.0 + epsilon,
        "Total probability should be one, but is {}.",
        sum
    );
}
