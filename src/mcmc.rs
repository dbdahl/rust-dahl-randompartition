use crate::clust::Clustering;
use crate::distr::FullConditional;
use crate::perm::Permutation;

use rand::prelude::*;

pub fn update_neal_algorithm3<T, U, V>(
    n_updates: u32,
    mut state: Clustering,
    permutation: &Permutation,
    prior: &T,
    log_posterior_predictive: &U,
    rng: &mut V,
) -> Clustering
where
    T: FullConditional,
    U: Fn(usize, &[usize]) -> f64,
    V: Rng,
{
    for _ in 0..n_updates {
        for i in 0..state.n_items() {
            let ii = permutation.get(i);
            let labels_and_log_weights =
                prior
                    .log_full_conditional(ii, &state)
                    .into_iter()
                    .map(|(label, log_prior)| {
                        let indices = &state.items_of_without(label, ii)[..];
                        (label, log_posterior_predictive(ii, indices) + log_prior)
                    });
            let pair = state.select(labels_and_log_weights, true, 0, Some(rng), false);
            state.allocate(ii, pair.0);
        }
    }
    state
}

pub fn update_neal_algorithm8<T, U, V>(
    n_updates: u32,
    mut state: Clustering,
    permutation: &Permutation,
    prior: &T,
    log_likelihood_contribution_fn: &U,
    rng: &mut V,
) -> Clustering
where
    T: FullConditional,
    U: Fn(usize, usize, bool) -> f64,
    V: Rng,
{
    for _ in 0..n_updates {
        for i in 0..state.n_items() {
            let ii = permutation.get(i);
            let labels_and_log_weights =
                prior
                    .log_full_conditional(ii, &state)
                    .into_iter()
                    .map(|(label, log_prior)| {
                        (
                            label,
                            log_likelihood_contribution_fn(ii, label, state.size_of(label) == 0)
                                + log_prior,
                        )
                    });
            let pair = state.select(labels_and_log_weights, true, 0, Some(rng), false);
            state.allocate(ii, pair.0);
        }
    }
    state
}

fn make_posterior<'a, T: 'a, U: 'a>(log_prior: T, log_likelihood: U) -> impl Fn(&Clustering) -> f64
where
    T: Fn(&Clustering) -> f64,
    U: Fn(&[usize]) -> f64,
{
    move |clustering: &Clustering| {
        clustering
            .allocation()
            .iter()
            .fold(log_prior(clustering), |sum, label| {
                sum + log_likelihood(&clustering.items_of(*label)[..])
            })
    }
}

#[cfg(test)]
mod tests_mcmc {
    use super::*;
    use crate::crp::CrpParameters;
    use crate::prelude::*;

    #[test]
    fn test_crp_neal_algorithm3() {
        let mut current = Clustering::one_cluster(5);
        let neal_functions = CrpParameters::new_with_mass(current.n_items(), Mass::new(1.0));
        let permutation = Permutation::natural_and_fixed(current.n_items());
        let log_posterior_predictive = |_i: usize, _indices: &[usize]| 0.0;
        let mut sum = 0;
        let n_samples = 10000;
        for _ in 0..n_samples {
            current = update_neal_algorithm3(
                2,
                current,
                &permutation,
                &neal_functions,
                &log_posterior_predictive,
                &mut thread_rng(),
            );
            sum += current.n_clusters();
        }
        let mean_number_of_subsets = (sum as f64) / (n_samples as f64);
        let z_stat = (mean_number_of_subsets - 2.283333) / (0.8197222 / n_samples as f64).sqrt();
        assert!(z_stat.abs() < 3.290527);
    }
}
