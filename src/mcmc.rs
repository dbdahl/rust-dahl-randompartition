use crate::clust::Clustering;
use crate::distr::{
    FullConditional, HasPermutation, HasScalarShrinkage, NormalizedProbabilityMassFunction,
    ProbabilityMassFunction,
};
use crate::perm::Permutation;
use crate::slice::slice_sampler;
use rand::Rng;
use statrs::distribution::{Continuous, Gamma};

pub fn update_neal_algorithm3<T, U, V>(
    n_updates: u32,
    state: &mut Clustering,
    permutation: &Permutation,
    prior: &T,
    log_posterior_predictive: &U,
    rng: &mut V,
) where
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
}

pub fn update_neal_algorithm8<T, U, V>(
    n_updates: u32,
    state: &mut Clustering,
    permutation: &Permutation,
    prior: &T,
    log_likelihood_contribution_fn: &mut U,
    rng: &mut V,
) where
    T: FullConditional,
    U: FnMut(usize, usize, bool) -> f64,
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
}

pub fn update_neal_algorithm_full<T, U, V>(
    n_updates: u32,
    state: &mut Clustering,
    permutation: &Permutation,
    prior: &T,
    log_likelihood_contribution_fn: &mut U,
    rng: &mut V,
) where
    T: FullConditional,
    U: FnMut(&Clustering) -> f64,
    V: Rng,
{
    for _ in 0..n_updates {
        for i in 0..state.n_items() {
            let ii = permutation.get(i);
            let labels_and_log_weights = prior
                .log_full_conditional(ii, &state)
                .into_iter()
                .map(|(label, log_prior)| {
                    state.allocate(ii, label);
                    (label, log_likelihood_contribution_fn(&state) + log_prior)
                })
                .collect::<Vec<_>>()
                .into_iter();
            let pair = state.select(labels_and_log_weights, true, 0, Some(rng), false);
            state.allocate(ii, pair.0);
        }
    }
}

pub fn update_permutation<T, V>(
    n_updates: u32,
    prior: &mut T,
    n_items_per_update: u32,
    clustering: &Clustering,
    rng: &mut V,
) -> u32
where
    T: ProbabilityMassFunction + NormalizedProbabilityMassFunction + HasPermutation,
    V: Rng,
{
    let n_items_per_update = n_items_per_update.try_into().unwrap();
    let mut n_acceptances = 0;
    let mut log_pmf_current = prior.log_pmf(clustering);
    for _ in 0..n_updates {
        prior
            .permutation_mut()
            .partial_shuffle(n_items_per_update, rng);
        let log_pmf_proposal = prior.log_pmf(clustering);
        let log_hastings_ratio = log_pmf_proposal - log_pmf_current;
        if 0.0 <= log_hastings_ratio || rng.gen_range(0.0..1.0_f64).ln() < log_hastings_ratio {
            n_acceptances += 1;
            log_pmf_current = log_pmf_proposal;
        } else {
            prior
                .permutation_mut()
                .partial_shuffle_undo(n_items_per_update);
        }
    }
    n_acceptances
}

pub fn update_scalar_shrinkage<T, V>(
    n_updates: u32,
    prior: &mut T,
    w: f64,
    shape: f64,
    rate: f64,
    clustering: &Clustering,
    rng: &mut V,
) -> u32
where
    T: ProbabilityMassFunction + NormalizedProbabilityMassFunction + HasScalarShrinkage,
    V: Rng,
{
    let gamma_distribution = Gamma::new(shape, rate).unwrap();
    for _ in 0..n_updates {
        let x = *prior.shrinkage();
        let f = |x| {
            *prior.shrinkage_mut() = x;
            prior.log_pmf(clustering) + gamma_distribution.ln_pdf(x)
        };
        let (x_new, _) = slice_sampler(x, f, w, u32::MAX, true, rng);
        *prior.shrinkage_mut() = x_new;
    }
    n_updates
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
    use rand::prelude::*;

    #[test]
    fn test_crp_neal_algorithm3() {
        let mut current = Clustering::one_cluster(5);
        let neal_functions = CrpParameters::new_with_mass(current.n_items(), Mass::new(1.0));
        let permutation = Permutation::natural_and_fixed(current.n_items());
        let log_posterior_predictive = |_i: usize, _indices: &[usize]| 0.0;
        let mut sum = 0;
        let n_samples = 10000;
        for _ in 0..n_samples {
            update_neal_algorithm3(
                2,
                &mut current,
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
