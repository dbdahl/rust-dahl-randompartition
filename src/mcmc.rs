use crate::clust::Clustering;
use crate::distr::{
    FullConditional, HasCost, HasPermutation, HasScalarShrinkage, HasVectorShrinkage,
    NormalizedProbabilityMassFunction, ProbabilityMassFunction,
};
use crate::perm::Permutation;
use crate::prelude::*;
use crate::slice::slice_sampler;
use rand::Rng;
use statrs::distribution::{Beta, Continuous, Gamma};

pub fn update_partition_gibbs<T, U, V>(
    n_updates: u32,
    state: &mut Clustering,
    permutation: &Permutation,
    prior: &T,
    log_likelihood: &mut U,
    rng: &mut V,
) where
    T: FullConditional,
    U: FnMut(usize, &Clustering) -> f64,
    V: Rng,
{
    for _ in 0..n_updates {
        for i in 0..state.n_items() {
            let ii = permutation.get(i);
            let labels_and_log_weights: Vec<_> = prior
                .log_full_conditional(ii, state)
                .into_iter()
                .map(|(label, log_prior)| {
                    state.allocate(ii, label);
                    (label, log_likelihood(ii, state) + log_prior)
                })
                .collect();
            let pair = state.select(
                labels_and_log_weights.into_iter(),
                true,
                false,
                0,
                Some(rng),
                false,
            );
            state.allocate(ii, pair.0);
        }
    }
}

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
                    .log_full_conditional(ii, state)
                    .into_iter()
                    .map(|(label, log_prior)| {
                        let indices = &state.items_of_without(label, ii)[..];
                        (label, log_posterior_predictive(ii, indices) + log_prior)
                    });
            let pair = state.select(labels_and_log_weights, true, false, 0, Some(rng), false);
            state.allocate(ii, pair.0);
        }
    }
}

pub fn update_neal_algorithm8<T, U, W, V, X>(
    n_updates: u32,
    state: &mut Clustering,
    permutation: &Permutation,
    prior: &T,
    log_likelihood_contribution_fn: &mut U,
    common_item_cacher: W,
    rng: &mut V,
) where
    T: FullConditional,
    U: FnMut(usize, &X, usize, bool) -> f64,
    W: Fn(usize) -> X,
    V: Rng,
    X: Sized,
{
    for _ in 0..n_updates {
        for i in 0..state.n_items() {
            let ii = permutation.get(i);
            let cache = common_item_cacher(ii);
            let labels_and_log_weights =
                prior
                    .log_full_conditional(ii, state)
                    .into_iter()
                    .map(|(label, log_prior)| {
                        (
                            label,
                            log_likelihood_contribution_fn(
                                ii,
                                &cache,
                                label,
                                state.size_of(label) == 0,
                            ) + log_prior,
                        )
                    });
            let pair = state.select(labels_and_log_weights, true, false, 0, Some(rng), false);
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
                .log_full_conditional(ii, state)
                .into_iter()
                .map(|(label, log_prior)| {
                    state.allocate(ii, label);
                    (label, log_likelihood_contribution_fn(state) + log_prior)
                })
                .collect::<Vec<_>>()
                .into_iter();
            let pair = state.select(labels_and_log_weights, true, false, 0, Some(rng), false);
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
    if n_items_per_update <= 1 {
        return 0;
    }
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
    shape: Shape,
    rate: Rate,
    clustering: &Clustering,
    rng: &mut V,
) -> u32
where
    T: ProbabilityMassFunction + NormalizedProbabilityMassFunction + HasScalarShrinkage,
    V: Rng,
{
    if w <= 0.0 {
        return 0;
    }
    let gamma_distribution = Gamma::new(shape.get(), rate.get()).unwrap();
    for _ in 0..n_updates {
        let x = prior.shrinkage().get();
        let f = |new_value| match prior.shrinkage_mut().set(new_value) {
            None => f64::NEG_INFINITY,
            _ => prior.log_pmf(clustering) + gamma_distribution.ln_pdf(new_value),
        };
        let (_x_new, _) = slice_sampler(x, f, w, u32::MAX, true, rng);
        // prior.shrinkage_mut().set(_x_new); // Not necessary... see implementation of slice_sampler function.
    }
    n_updates
}

#[allow(clippy::too_many_arguments)]
pub fn update_vector_shrinkage<T, V>(
    n_updates: u32,
    prior: &mut T,
    reference: usize,
    w: f64,
    shape: Shape,
    rate: Rate,
    clustering: &Clustering,
    rng: &mut V,
) -> u32
where
    T: ProbabilityMassFunction + NormalizedProbabilityMassFunction + HasVectorShrinkage,
    V: Rng,
{
    if w <= 0.0 {
        return 0;
    }
    let gamma_distribution = Gamma::new(shape.get(), rate.get()).unwrap();
    for _ in 0..n_updates {
        let x = prior.shrinkage()[reference];
        let f = |new_value: f64| match ScalarShrinkage::new(new_value) {
            None => f64::NEG_INFINITY,
            Some(new_value) => {
                prior
                    .shrinkage_mut()
                    .rescale_by_reference(reference, new_value);
                prior.log_pmf(clustering) + gamma_distribution.ln_pdf(new_value.get())
            }
        };
        let (_x_new, _) = slice_sampler(x.get(), f, w, 100, true, rng);
        // prior.shrinkage_mut().rescale_by_reference(reference, _x_new); // Not necessary... see implementation of slice_sampler function.
    }
    n_updates
}

pub fn update_cost<T, V>(
    n_updates: u32,
    prior: &mut T,
    w: f64,
    shape1: Shape,
    shape2: Shape,
    clustering: &Clustering,
    rng: &mut V,
) -> u32
where
    T: ProbabilityMassFunction + NormalizedProbabilityMassFunction + HasCost,
    V: Rng,
{
    if w <= 0.0 {
        return 0;
    }
    let beta_distribution = Beta::new(shape1.get(), shape2.get()).unwrap();
    for _ in 0..n_updates {
        let x = prior.cost().get();
        let f = |new_value| match prior.cost_mut().set(new_value) {
            None => f64::NEG_INFINITY,
            _ => prior.log_pmf(clustering) + beta_distribution.ln_pdf(new_value / 2.0),
        };
        let (_x_new, _) = slice_sampler(x, f, w, u32::MAX, true, rng);
        // prior.shrinkage_mut().set(_x_new); // Not necessary... see implementation of slice_sampler function.
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
    use rand::prelude::*;

    #[test]
    fn test_crp_neal_algorithm3() {
        let mut current = Clustering::one_cluster(5);
        let neal_functions =
            CrpParameters::new(current.n_items(), Concentration::new(1.0).unwrap());
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
