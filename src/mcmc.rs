use crate::clust::Clustering;
use crate::distr::{FullConditional, PredictiveProbabilityFunction};
use crate::perm::Permutation;
use crate::push_into_slice_i32;

//use crate::frp;
use crate::cpp::CppParameters;
use crate::crp::CrpParameters;
use crate::epa::EpaParameters;
use crate::frp::FrpParameters;
use crate::lsp::LspParameters;
use crate::sp::SpParameters;
use crate::up::UpParameters;
use dahl_roxido::mk_rng_isaac;
use rand::prelude::*;
use std::ffi::c_void;
use std::slice;

pub fn update_neal_algorithm3<T, U, V>(
    n_updates: u32,
    current: &Clustering,
    permutation: &Permutation,
    prior: &T,
    log_posterior_predictive: &U,
    rng: &mut V,
) -> Clustering
where
    T: PredictiveProbabilityFunction,
    U: Fn(usize, &[usize]) -> f64,
    V: Rng,
{
    let mut state = current.clone();
    for _ in 0..n_updates {
        for i in 0..state.n_items() {
            let ii = permutation.get(i);
            let labels_and_weights = state.available_labels_for_reallocation(ii).map(|label| {
                let indices = &state.items_of_without(label, ii)[..];
                let log_weight = log_posterior_predictive(ii, indices)
                    + prior.log_predictive_probability(ii, label, &state);
                (label, log_weight)
            });
            let pair = state.select(labels_and_weights, true, 0, Some(rng), false);
            state.reallocate(ii, pair.0);
        }
    }
    state
}

pub fn update_neal_algorithm8<T, U, V>(
    n_updates: u32,
    current: &Clustering,
    permutation: &Permutation,
    prior: &T,
    log_likelihood_contribution_fn: &U,
    rng: &mut V,
) -> Clustering
where
    T: PredictiveProbabilityFunction,
    U: Fn(usize, usize, bool) -> f64,
    V: Rng,
{
    let mut state = current.clone();
    for _ in 0..n_updates {
        for i in 0..state.n_items() {
            let ii = permutation.get(i);
            let labels_and_weights = state.available_labels_for_reallocation(ii).map(|label| {
                let log_weight =
                    log_likelihood_contribution_fn(ii, label, state.size_of(label) == 0)
                        + prior.log_predictive_probability(ii, label, &state);
                (label, log_weight)
            });
            let pair = state.select(labels_and_weights, true, 0, Some(rng), false);
            state.reallocate(ii, pair.0);
        }
    }
    state
}

pub fn update_neal_algorithm8_v2<T, U, V>(
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
            state.reallocate(ii, pair.0);
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
    use crate::prelude::*;

    #[test]
    fn test_crp_neal_algorithm3() {
        let mut current = Clustering::one_cluster(5);
        let neal_functions = CrpParameters::new_with_mass(Mass::new(1.0), current.n_items());
        let permutation = Permutation::natural_and_fixed(current.n_items());
        let log_posterior_predictive = |_i: usize, _indices: &[usize]| 0.0;
        let mut sum = 0;
        let n_samples = 10000;
        for _ in 0..n_samples {
            current = update_neal_algorithm3(
                2,
                &current,
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

extern "C" {
    fn rrAllocVectorINTSXP(len: i32) -> Rr_Sexp_Vector_Intsxp;
    fn callRFunction_logIntegratedLikelihoodItem(
        fn_ptr: *const c_void,
        i: i32,
        indices: Rr_Sexp_Vector_Intsxp,
        env_ptr: *const c_void,
    ) -> f64;
    fn callRFunction_logIntegratedLikelihoodSubset(
        fn_ptr: *const c_void,
        indices: Rr_Sexp_Vector_Intsxp,
        env_ptr: *const c_void,
    ) -> f64;
    fn callRFunction_logLikelihoodItem(
        fn_ptr: *const c_void,
        i: i32,
        label: i32,
        is_new: i32,
        env_ptr: *const c_void,
    ) -> f64;
}

#[repr(C)]
pub struct Rr_Sexp {
    pub sexp_ptr: *const c_void,
}

#[repr(C)]
pub struct Rr_Sexp_Vector_Intsxp {
    pub sexp_ptr: *const c_void,
    pub data_ptr: *mut i32,
    pub len: i32,
}

impl Rr_Sexp_Vector_Intsxp {
    fn from_slice_offset_by_1(slice: &[usize]) -> Self {
        let result = unsafe { rrAllocVectorINTSXP(slice.len() as i32) };
        let into_slice: &mut [i32] =
            unsafe { slice::from_raw_parts_mut(result.data_ptr, result.len as usize) };
        for (x, y) in into_slice.iter_mut().zip(slice) {
            *x = (*y as i32) + 1;
        }
        result
    }
    fn from_slice(slice: &[usize]) -> Self {
        let result = unsafe { rrAllocVectorINTSXP(slice.len() as i32) };
        let into_slice: &mut [i32] =
            unsafe { slice::from_raw_parts_mut(result.data_ptr, result.len as usize) };
        for (x, y) in into_slice.iter_mut().zip(slice) {
            *x = *y as i32;
        }
        result
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__neal_algorithm3(
    n_updates_for_partition: i32,
    n_items: i32,
    partition_ptr: *mut i32,
    prior_only: i32,
    log_posterior_predictive_function_ptr: *const c_void,
    env_ptr: *const c_void,
    seed_ptr: *const i32, // Assumed length is 32
    prior_id: i32,
    prior_ptr: *const c_void,
) {
    let nup = n_updates_for_partition as u32;
    let ni = n_items as usize;
    let clustering_slice = slice::from_raw_parts_mut(partition_ptr, ni);
    let current = Clustering::from_slice(clustering_slice);
    let log_like: Box<dyn Fn(usize, &[usize]) -> f64> = if prior_only != 0 {
        Box::new(|_i: usize, _indices: &[usize]| 0.0)
    } else {
        Box::new(move |i: usize, indices: &[usize]| {
            callRFunction_logIntegratedLikelihoodItem(
                log_posterior_predictive_function_ptr,
                (i as i32) + 1,
                Rr_Sexp_Vector_Intsxp::from_slice_offset_by_1(indices),
                env_ptr,
            )
        })
    };
    let perm = Permutation::natural_and_fixed(current.n_items());
    let mut rng = mk_rng_isaac(seed_ptr);
    let mut clustering = match prior_id {
        0 => current,
        1 => {
            let p = std::ptr::NonNull::new(prior_ptr as *mut CrpParameters).unwrap();
            update_neal_algorithm3(nup, &current, &perm, p.as_ref(), &log_like, &mut rng)
        }
        2 => {
            let p = std::ptr::NonNull::new(prior_ptr as *mut FrpParameters).unwrap();
            update_neal_algorithm3(nup, &current, &perm, p.as_ref(), &log_like, &mut rng)
        }
        3 => {
            let p = std::ptr::NonNull::new(prior_ptr as *mut LspParameters).unwrap();
            update_neal_algorithm3(nup, &current, &perm, p.as_ref(), &log_like, &mut rng)
        }
        4 => {
            let p = std::ptr::NonNull::new(prior_ptr as *mut CppParameters).unwrap();
            update_neal_algorithm3(nup, &current, &perm, p.as_ref(), &log_like, &mut rng)
        }
        5 => {
            let p = std::ptr::NonNull::new(prior_ptr as *mut EpaParameters).unwrap();
            update_neal_algorithm3(nup, &current, &perm, p.as_ref(), &log_like, &mut rng)
        }
        6 => {
            let p = std::ptr::NonNull::new(prior_ptr as *mut SpParameters).unwrap();
            update_neal_algorithm3(nup, &current, &perm, p.as_ref(), &log_like, &mut rng)
        }
        7 => {
            let p = std::ptr::NonNull::new(prior_ptr as *mut UpParameters).unwrap();
            update_neal_algorithm3(nup, &current, &perm, p.as_ref(), &log_like, &mut rng)
        }
        _ => panic!("Unsupported prior ID: {}", prior_id),
    };
    clustering = clustering.relabel(1, None, false).0;
    push_into_slice_i32(&clustering.allocation()[..], clustering_slice);
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__neal_algorithm8(
    n_updates_for_partition: i32,
    n_items: i32,
    partition_ptr: *mut i32,
    prior_only: i32,
    log_likelihood_function_ptr: *const c_void,
    env_ptr: *const c_void,
    seed_ptr: *const i32, // Assumed length is 32
    prior_id: i32,
    prior_ptr: *const c_void,
    map_ptr: &mut Rr_Sexp,
) {
    let nup = n_updates_for_partition as u32;
    let ni = n_items as usize;
    let clustering_slice = slice::from_raw_parts_mut(partition_ptr, ni);
    let mut current = Clustering::from_slice(clustering_slice);
    current.exclude_label(0);
    let log_like: Box<dyn Fn(usize, usize, bool) -> f64> = if prior_only != 0 {
        Box::new(|_i: usize, _label: usize, _is_new: bool| 0.0)
    } else {
        Box::new(move |i: usize, label: usize, is_new: bool| {
            callRFunction_logLikelihoodItem(
                log_likelihood_function_ptr,
                (i + 1) as i32,
                label as i32,
                is_new as i32,
                env_ptr,
            )
        })
    };
    let perm = Permutation::natural_and_fixed(current.n_items());
    let mut rng = mk_rng_isaac(seed_ptr);
    current = match prior_id {
        0 => current,
        1 => {
            let p = std::ptr::NonNull::new(prior_ptr as *mut CrpParameters).unwrap();
            update_neal_algorithm8_v2(nup, current, &perm, p.as_ref(), &log_like, &mut rng)
        }
        2 => {
            let p = std::ptr::NonNull::new(prior_ptr as *mut FrpParameters).unwrap();
            update_neal_algorithm8(nup, &current, &perm, p.as_ref(), &log_like, &mut rng)
        }
        3 => {
            let p = std::ptr::NonNull::new(prior_ptr as *mut LspParameters).unwrap();
            update_neal_algorithm8(nup, &current, &perm, p.as_ref(), &log_like, &mut rng)
        }
        4 => {
            let p = std::ptr::NonNull::new(prior_ptr as *mut CppParameters).unwrap();
            update_neal_algorithm8(nup, &current, &perm, p.as_ref(), &log_like, &mut rng)
        }
        5 => {
            let p = std::ptr::NonNull::new(prior_ptr as *mut EpaParameters).unwrap();
            update_neal_algorithm8(nup, &current, &perm, p.as_ref(), &log_like, &mut rng)
        }
        6 => {
            let p = std::ptr::NonNull::new(prior_ptr as *mut SpParameters).unwrap();
            update_neal_algorithm8(nup, &current, &perm, p.as_ref(), &log_like, &mut rng)
        }
        7 => {
            let p = std::ptr::NonNull::new(prior_ptr as *mut UpParameters).unwrap();
            update_neal_algorithm8_v2(nup, current, &perm, p.as_ref(), &log_like, &mut rng)
        }
        _ => panic!("Unsupported prior ID: {}", prior_id),
    };
    let (clustering, map) = current.relabel(1, None, true);
    push_into_slice_i32(&clustering.allocation()[..], clustering_slice);
    map_ptr.sexp_ptr = Rr_Sexp_Vector_Intsxp::from_slice(&map.unwrap()[1..]).sexp_ptr;
}
