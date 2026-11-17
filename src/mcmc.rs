use crate::clust::Clustering;
use crate::distr::{
    FullConditional, HasGrit, HasPermutation, HasScalarShrinkage, HasVectorShrinkage,
    NormalizedProbabilityMassFunction, ProbabilityMassFunction,
};
use crate::perm::Permutation;
use crate::prelude::*;
use rand::{Rng, RngExt};
use slice_sampler::univariate::stepping_out::{
    TuningParameters, univariate_slice_sampler_stepping_out_and_shrinkage as slice_sampler,
};
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
        if 0.0 <= log_hastings_ratio || rng.random_range(0.0..1.0_f64).ln() < log_hastings_ratio {
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

// The slice updates below give their targets the support of their priors:
// positive, finite shrinkage for a gamma prior, and grit strictly between 0 and
// 1 for a beta prior. Each must start inside it, since the slice sampler needs
// a starting point with a positive, finite density, and its target stays inside
// it, so that it never sets a value outside the support. A shrinkage of 0 for
// the reference item, in particular, would leave nothing to rescale.

fn in_gamma_support(x: f64) -> bool {
    x > 0.0 && x.is_finite()
}

fn in_beta_support(x: f64) -> bool {
    0.0 < x && x < 1.0
}

// The slice sampler's tuning for the width 'w', or None to skip the update, as
// for a width that isn't positive.
fn tuning_parameters(w: f64) -> Option<TuningParameters> {
    if w <= 0.0 {
        return None;
    }
    assert!(
        w.is_finite(),
        "the slice sampler's width is {w}, but it must be finite, or not positive to skip the update"
    );
    Some(TuningParameters::new().width(w))
}

/// Updates the shrinkage with a slice sampler `n_updates` times, under a gamma
/// prior with `shape` and `rate`, unless the width `w` isn't positive.
///
/// # Panics
///
/// If `w` is NaN or infinite, or if the shrinkage isn't positive and finite.
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
    let Some(tuning_parameters) = tuning_parameters(w) else {
        return 0;
    };
    let x = prior.shrinkage().get();
    assert!(
        in_gamma_support(x),
        "the shrinkage is {x}, but it must be positive and finite to update it under a gamma prior"
    );
    let rng = &mut Some(fastrand::Rng::with_seed(rng.next_u64()));
    let gamma_distribution = Gamma::new(shape.get(), rate.get()).unwrap();
    for _ in 0..n_updates {
        let x = prior.shrinkage().get();
        let f = |new_value: f64| {
            if !in_gamma_support(new_value) {
                return f64::NEG_INFINITY;
            }
            *prior.shrinkage_mut() = ScalarShrinkage::new_unchecked(new_value);
            prior.log_pmf(clustering) + gamma_distribution.ln_pdf(new_value)
        };
        // The slice sampler last evaluates f at the value it returns, which so
        // remains set.
        slice_sampler(x, f, true, &tuning_parameters, rng);
    }
    n_updates
}

/// Updates the shrinkage with a slice sampler `n_updates` times, rescaling every
/// item's so that the `reference` item's has a gamma prior with `shape` and
/// `rate`, unless the width `w` isn't positive.
///
/// # Panics
///
/// If `w` is NaN or infinite, if `reference` isn't an item, or if the
/// reference item's shrinkage isn't positive and finite.
#[allow(clippy::too_many_arguments)]
pub fn update_shrinkage<T, V>(
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
    let Some(tuning_parameters) = tuning_parameters(w) else {
        return 0;
    };
    let n_items = prior.shrinkage().n_items();
    assert!(
        reference < n_items,
        "the reference item is {reference}, but there are {n_items} items"
    );
    let x = prior.shrinkage()[reference].get();
    assert!(
        in_gamma_support(x),
        "the reference item's shrinkage is {x}, but it must be positive and finite to update the shrinkage under a gamma prior"
    );
    let rng = &mut Some(fastrand::Rng::with_seed(rng.next_u64()));
    let gamma_distribution = Gamma::new(shape.get(), rate.get()).unwrap();
    for _ in 0..n_updates {
        let x = prior.shrinkage()[reference].get();
        let f = |new_value: f64| {
            if !in_gamma_support(new_value) {
                return f64::NEG_INFINITY;
            }
            prior
                .shrinkage_mut()
                .rescale_by_reference(reference, ScalarShrinkage::new_unchecked(new_value));
            prior.log_pmf(clustering) + gamma_distribution.ln_pdf(new_value)
        };
        // The slice sampler last evaluates f at the value it returns, which so
        // remains set.
        slice_sampler(x, f, true, &tuning_parameters, rng);
    }
    n_updates
}

/// Updates the grit with a slice sampler `n_updates` times, under a beta prior
/// with `shape1` and `shape2`, unless the width `w` isn't positive.
///
/// # Panics
///
/// If `w` is NaN or infinite, or if the grit isn't strictly between 0 and 1.
pub fn update_grit<T, V>(
    n_updates: u32,
    prior: &mut T,
    w: f64,
    shape1: Shape,
    shape2: Shape,
    clustering: &Clustering,
    rng: &mut V,
) -> u32
where
    T: ProbabilityMassFunction + NormalizedProbabilityMassFunction + HasGrit,
    V: Rng,
{
    let Some(tuning_parameters) = tuning_parameters(w) else {
        return 0;
    };
    let x = prior.grit().get();
    assert!(
        in_beta_support(x),
        "the grit is {x}, but it must be strictly between 0 and 1 to update it under a beta prior"
    );
    let rng = &mut Some(fastrand::Rng::with_seed(rng.next_u64()));
    let beta_distribution = Beta::new(shape1.get(), shape2.get()).unwrap();
    for _ in 0..n_updates {
        let x = prior.grit().get();
        let f = |new_value: f64| {
            if !in_beta_support(new_value) {
                return f64::NEG_INFINITY;
            }
            *prior.grit_mut() = Grit::new_unchecked(new_value);
            prior.log_pmf(clustering) + beta_distribution.ln_pdf(new_value)
        };
        // The slice sampler last evaluates f at the value it returns, which so
        // remains set.
        slice_sampler(x, f, true, &tuning_parameters, rng);
    }
    n_updates
}

fn make_posterior<T, U>(log_prior: T, log_likelihood: U) -> impl Fn(&Clustering) -> f64
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
                &mut rand::rng(),
            );
            sum += current.n_clusters();
        }
        let mean_number_of_subsets = (sum as f64) / (n_samples as f64);
        let z_stat = (mean_number_of_subsets - 2.283333) / (0.8197222 / n_samples as f64).sqrt();
        assert!(z_stat.abs() < 3.290527);
    }

    // The slice updates' starting points and widths with which they once never
    // ended. If a check regresses, its test hangs rather than fails.

    use crate::lsp::LspParameters;
    use crate::shrink::Shrinkage;
    use crate::sp::SpParameters;
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    fn anchor() -> Clustering {
        Clustering::from_vector(vec![0, 0, 0, 1, 1, 2, 2, 2])
    }

    fn sp(shrinkage: &[f64], grit: f64) -> SpParameters<CrpParameters> {
        let n_items = shrinkage.len();
        SpParameters::new(
            anchor(),
            Shrinkage::from(shrinkage).unwrap(),
            Permutation::natural(n_items),
            Grit::new(grit).unwrap(),
            CrpParameters::new(n_items, Concentration::new(1.0).unwrap()),
            true,
        )
        .unwrap()
    }

    fn lsp(shrinkage: f64) -> LspParameters {
        LspParameters::new_with_shrinkage(
            anchor(),
            ScalarShrinkage::new(shrinkage).unwrap(),
            Concentration::new(1.0).unwrap(),
            Permutation::natural(8),
        )
        .unwrap()
    }

    fn update_sp_shrinkage(pd: &mut SpParameters<CrpParameters>, reference: usize, w: f64) -> u32 {
        let shape = Shape::new(2.0).unwrap();
        let rate = Rate::new(1.0).unwrap();
        let mut rng = Pcg64Mcg::seed_from_u64(1);
        update_shrinkage(50, pd, reference, w, shape, rate, &anchor(), &mut rng)
    }

    fn update_sp_grit(pd: &mut SpParameters<CrpParameters>) {
        let shape = Shape::new(2.0).unwrap();
        let mut rng = Pcg64Mcg::seed_from_u64(1);
        update_grit(50, pd, 0.5, shape, shape, &anchor(), &mut rng);
    }

    fn update_lsp_shrinkage(pd: &mut LspParameters) {
        let shape = Shape::new(2.0).unwrap();
        let rate = Rate::new(1.0).unwrap();
        let mut rng = Pcg64Mcg::seed_from_u64(1);
        update_scalar_shrinkage(50, pd, 1.0, shape, rate, &anchor(), &mut rng);
    }

    #[test]
    fn shrinkage_updates_keep_items_in_proportion() {
        let start = [0.0, 0.0, 0.0, 3.0, 3.0, 1.0, 1.0, 1.0];
        let mut pd = sp(&start, 0.5);
        assert_eq!(update_sp_shrinkage(&mut pd, 3, 1.0), 50);
        let shrinkage: Vec<_> = pd.shrinkage.as_slice().iter().map(|x| x.get()).collect();
        let scale = shrinkage[3] / 3.0;
        assert!(scale > 0.0 && scale.is_finite());
        for (x, y) in shrinkage.iter().zip(start) {
            assert!((x - scale * y).abs() <= 1e-12 * x.max(1.0));
        }
    }

    #[test]
    fn a_width_that_is_not_positive_skips_the_update() {
        let mut pd = sp(&[1.0; 8], 0.5);
        assert_eq!(update_sp_shrinkage(&mut pd, 0, 0.0), 0);
        assert_eq!(update_sp_shrinkage(&mut pd, 0, -1.0), 0);
        assert!(pd.shrinkage.as_slice().iter().all(|x| x.get() == 1.0));
    }

    #[test]
    #[should_panic(expected = "the slice sampler's width is NaN")]
    fn a_nan_width_panics() {
        update_sp_shrinkage(&mut sp(&[1.0; 8], 0.5), 0, f64::NAN);
    }

    #[test]
    #[should_panic(expected = "the reference item's shrinkage is 0, but it must be positive")]
    fn a_reference_shrinkage_of_0_panics() {
        update_sp_shrinkage(
            &mut sp(&[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.5),
            0,
            1.0,
        );
    }

    #[test]
    #[should_panic(expected = "the reference item's shrinkage is inf, but it must be positive")]
    fn an_infinite_reference_shrinkage_panics() {
        let mut shrinkage = [1.0; 8];
        shrinkage[2] = f64::INFINITY;
        update_sp_shrinkage(&mut sp(&shrinkage, 0.5), 2, 1.0);
    }

    #[test]
    #[should_panic(expected = "the reference item is 8, but there are 8 items")]
    fn a_reference_that_is_not_an_item_panics() {
        update_sp_shrinkage(&mut sp(&[1.0; 8], 0.5), 8, 1.0);
    }

    #[test]
    fn grit_updates_stay_between_0_and_1() {
        let mut pd = sp(&[1.0; 8], 0.5);
        update_sp_grit(&mut pd);
        assert!(0.0 < pd.grit.get() && pd.grit.get() < 1.0);
    }

    #[test]
    #[should_panic(expected = "the grit is 1.5, but it must be strictly between 0 and 1")]
    fn a_grit_above_1_panics() {
        update_sp_grit(&mut sp(&[1.0; 8], 1.5));
    }

    #[test]
    #[should_panic(expected = "the grit is 0, but it must be strictly between 0 and 1")]
    fn a_grit_of_0_panics() {
        update_sp_grit(&mut sp(&[1.0; 8], 0.0));
    }

    #[test]
    fn scalar_shrinkage_updates_stay_positive() {
        let mut pd = lsp(2.0);
        update_lsp_shrinkage(&mut pd);
        let x = HasScalarShrinkage::shrinkage(&pd).get();
        assert!(x > 0.0 && x.is_finite());
    }

    #[test]
    #[should_panic(expected = "the shrinkage is 0, but it must be positive and finite")]
    fn a_scalar_shrinkage_of_0_panics() {
        update_lsp_shrinkage(&mut lsp(0.0));
    }

    #[test]
    #[should_panic(expected = "rescaling the shrinkage from the reference item's 0 to 2")]
    fn rescaling_from_0_panics() {
        let mut shrinkage = Shrinkage::from(&[0.0, 1.0]).unwrap();
        shrinkage.rescale_by_reference(0, ScalarShrinkage::new(2.0).unwrap());
    }

    #[test]
    #[should_panic(expected = "rescaling the shrinkage from the reference item's 1 to 0")]
    fn rescaling_to_0_panics() {
        let mut shrinkage = Shrinkage::from(&[1.0, 1.0]).unwrap();
        shrinkage.rescale_by_reference(0, ScalarShrinkage::zero());
    }
}
