use crate::frp;
use crate::prelude::*;
use crate::*;

use crate::crp::CRPParameters;
use crate::frp::{FRPParameters, Weights};
use dahl_partition::*;
use dahl_roxido::mk_rng_isaac;
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::*;
use rand_isaac::IsaacRng;
use std::ffi::c_void;
use std::slice;

pub trait NealFunctions {
    fn new_weight(&self, n_subsets: usize) -> f64;
    fn existing_weight(&self, n_subsets: usize, n_items: usize) -> f64;
}

pub fn update_neal_algorithm3<T, U, V>(
    n_updates: u32,
    current: &Partition,
    neal_functions: &T,
    log_posterior_predictive: &U,
    rng: &mut V,
) -> Partition
where
    T: NealFunctions,
    U: Fn(usize, &[usize]) -> f64,
    V: Rng,
{
    let ni = current.n_items();
    let mut state = current.clone();
    state.canonicalize();
    state.new_subset();
    let mut empty_subset_at_end = true;
    for _ in 0..n_updates {
        for i in 0..ni {
            // Remove 'i', ensure there is one and only one empty subset, and be efficient.
            let k = state.label_of(i).unwrap();
            state.remove(i);
            state.clean_subset(k);
            if state.subsets()[k].is_empty() {
                if empty_subset_at_end {
                    state.pop_subset();
                    empty_subset_at_end = k == state.n_subsets() - 1;
                } else {
                    state.canonicalize();
                    state.new_subset();
                    empty_subset_at_end = true;
                }
            }
            let n_subsets = state.n_subsets() - 1; // Because there is an empty subset
            let weights = state.subsets().iter().map(|subset| {
                let pp = if subset.is_empty() {
                    neal_functions.new_weight(n_subsets)
                } else {
                    neal_functions.existing_weight(n_subsets, subset.n_items())
                };
                log_posterior_predictive(i, &subset.items()[..]).exp() * pp
            });
            let dist = WeightedIndex::new(weights).unwrap();
            let subset_index = dist.sample(rng);
            state.add_with_index(i, subset_index);
            if state.subsets()[subset_index].n_items() == 1 {
                state.new_subset();
                empty_subset_at_end = true;
            }
        }
    }
    state.canonicalize();
    state
}

pub trait NealFunctionsGeneral {
    fn new_weight(&self, n_subsets: usize) -> f64;
    fn existing_weight(&self, index_index: usize, subset: &Subset, partition: &Partition) -> f64;
}

pub fn update_neal_algorithm3_generalized<T, U, V>(
    n_updates: u32,
    current: &Partition,
    permutation: &Permutation,
    neal_functions_generalized: &T,
    log_posterior_predictive: &U,
    rng: &mut V,
) -> Partition
where
    T: NealFunctionsGeneral,
    U: Fn(usize, &[usize]) -> f64,
    V: Rng,
{
    let ni = current.n_items();
    let mut state = current.clone();
    state.canonicalize();
    state.new_subset();
    let mut empty_subset_at_end = true;
    for _ in 0..n_updates {
        for i in 0..ni {
            let ii = permutation[i];
            // Remove 'i', ensure there is one and only one empty subset, and be efficient.
            let k = state.label_of(ii).unwrap();
            state.remove(ii);
            state.clean_subset(k);
            if state.subsets()[k].is_empty() {
                if empty_subset_at_end {
                    state.pop_subset();
                    empty_subset_at_end = k == state.n_subsets() - 1;
                } else {
                    state.canonicalize();
                    state.new_subset();
                    empty_subset_at_end = true;
                }
            }
            let n_subsets = state.n_subsets() - 1; // Because there is an empty subset
            let weights = state.subsets().iter().map(|subset| {
                let pp = if subset.is_empty() {
                    neal_functions_generalized.new_weight(n_subsets)
                } else {
                    neal_functions_generalized.existing_weight(i, &subset, &state)
                };
                log_posterior_predictive(ii, &subset.items()[..]).exp() * pp
            });
            let dist = WeightedIndex::new(weights).unwrap();
            let subset_index = dist.sample(rng);
            state.add_with_index(ii, subset_index);
            if state.subsets()[subset_index].n_items() == 1 {
                state.new_subset();
                empty_subset_at_end = true;
            }
        }
    }
    state.canonicalize();
    state
}

pub fn update_rwmh<T, U>(
    n_attempts: u32,
    current: &Partition,
    rate: Rate,
    mass: Mass,
    log_target: &T,
    rng: &mut U,
) -> (Partition, u32)
where
    T: Fn(&Partition) -> f64,
    U: Rng,
{
    let mut accepts: u32 = 0;
    let mut state = current.clone();
    let mut permutation = Permutation::natural(state.n_items());
    let mut log_target_state = log_target(&state);
    let mut weights_state = frp::Weights::constant(rate.unwrap(), state.n_subsets());
    for _ in 0..n_attempts {
        permutation.shuffle(rng);
        let current_parameters =
            FRPParameters::new(&state, &weights_state, &permutation, mass).unwrap();
        let proposal = frp::engine(&current_parameters, TargetOrRandom::Random(rng));
        let weights_proposal = frp::Weights::constant(rate.unwrap(), proposal.0.n_subsets());
        let log_target_proposal = log_target(&proposal.0);
        let log_ratio_target = log_target_proposal - log_target_state;
        let proposed_parameters =
            FRPParameters::new(&proposal.0, &weights_proposal, &permutation, mass).unwrap();
        let log_ratio_proposal = frp::engine(
            &proposed_parameters,
            TargetOrRandom::Target::<IsaacRng>(&mut state),
        )
        .1 - proposal.1;
        let log_mh_ratio = log_ratio_target + log_ratio_proposal;
        if log_mh_ratio >= 1.0 || rng.gen_range(0.0, 1.0) < log_mh_ratio.exp() {
            accepts += 1;
            state = proposal.0;
            log_target_state = log_target_proposal;
            weights_state = weights_proposal;
        };
    }
    state.canonicalize();
    (state, accepts)
}

fn make_posterior<'a, T: 'a, U: 'a>(log_prior: T, log_likelihood: U) -> impl Fn(&Partition) -> f64
where
    T: Fn(&Partition) -> f64,
    U: Fn(&[usize]) -> f64,
{
    move |partition: &Partition| {
        partition
            .subsets()
            .iter()
            .fold(log_prior(partition), |sum, subset| {
                sum + log_likelihood(&subset.items()[..])
            })
    }
}

#[cfg(test)]
mod tests_mcmc {
    use super::*;

    #[test]
    fn test_crp_rwmh() {
        let n_items = 5;
        let mut current = Partition::one_subset(n_items);
        let rate = Rate::new(5.0);
        let mass = Mass::new(1.0);
        let parameters = CRPParameters::new(mass);
        let log_prior = |p: &Partition| crate::crp::log_pmf(&p, &parameters);
        let log_likelihood = |_indices: &[usize]| 0.0;
        let log_target = make_posterior(log_prior, log_likelihood);
        let mut sum = 0;
        let n_samples = 10000;
        for _ in 0..n_samples {
            let result = update_rwmh(2, &current, rate, mass, &log_target, &mut thread_rng());
            current = result.0;
            sum += current.n_subsets();
        }
        let mean_number_of_subsets = (sum as f64) / (n_samples as f64);
        let z_stat = (mean_number_of_subsets - 2.283333) / (0.8197222 / n_samples as f64).sqrt();
        assert!(z_stat.abs() < 3.290527);
    }

    #[test]
    fn test_crp_neal_algorithm3() {
        let n_items = 5;
        let mut current = Partition::one_subset(n_items);
        let neal_functions = crp::CRPParameters::new(Mass::new(1.0));
        let log_posterior_predictive = |_i: usize, _indices: &[usize]| 0.0;
        let mut sum = 0;
        let n_samples = 10000;
        for _ in 0..n_samples {
            current = update_neal_algorithm3(
                2,
                &current,
                &neal_functions,
                &log_posterior_predictive,
                &mut thread_rng(),
            );
            sum += current.n_subsets();
        }
        let mean_number_of_subsets = (sum as f64) / (n_samples as f64);
        let z_stat = (mean_number_of_subsets - 2.283333) / (0.8197222 / n_samples as f64).sqrt();
        assert!(z_stat.abs() < 3.290527);
    }
}

extern "C" {
    fn callRFunction_logPosteriorPredictiveOfItem(
        fn_ptr: *const c_void,
        i: i32,
        indices: RR_SEXP_vector_INTSXP,
        env_ptr: *const c_void,
    ) -> f64;
    fn callRFunction_logIntegratedLikelihoodOfSubset(
        fn_ptr: *const c_void,
        indices: RR_SEXP_vector_INTSXP,
        env_ptr: *const c_void,
    ) -> f64;
    fn rrAllocVectorINTSXP(len: i32) -> RR_SEXP_vector_INTSXP;
}

#[repr(C)]
pub struct RR_SEXP_vector_INTSXP {
    pub sexp_ptr: *const c_void,
    pub data_ptr: *mut i32,
    pub len: i32,
}

impl RR_SEXP_vector_INTSXP {
    fn from_slice(slice: &[usize]) -> Self {
        let result = unsafe { rrAllocVectorINTSXP(slice.len() as i32) };
        let into_slice: &mut [i32] =
            unsafe { slice::from_raw_parts_mut(result.data_ptr, result.len as usize) };
        for (x, y) in into_slice.iter_mut().zip(slice) {
            *x = (*y as i32) + 1;
        }
        result
    }
}

unsafe fn neal_algorithm3_process_arguments<'a, 'b>(
    n_updates_for_partition: i32,
    n_items: i32,
    partition_ptr: *mut i32,
    prior_only: i32,
    log_posterior_predictive_function_ptr: *const c_void,
    env_ptr: *const c_void,
    seed_ptr: *const i32, // Assumed length is 32)
) -> (
    u32,
    &'a mut [i32],
    Partition,
    impl Fn(usize, &[usize]) -> f64,
    IsaacRng,
) {
    let nup = n_updates_for_partition as u32;
    let ni = n_items as usize;
    let partition_slice = slice::from_raw_parts_mut(partition_ptr, ni);
    let partition = Partition::from(partition_slice);
    let log_posterior_predictive: Box<dyn Fn(usize, &[usize]) -> f64> = if prior_only != 0 {
        Box::new(|_i: usize, _indices: &[usize]| 0.0)
    } else {
        Box::new(move |i: usize, indices: &[usize]| {
            callRFunction_logPosteriorPredictiveOfItem(
                log_posterior_predictive_function_ptr,
                (i as i32) + 1,
                RR_SEXP_vector_INTSXP::from_slice(indices),
                env_ptr,
            )
        })
    };
    let rng = mk_rng_isaac(seed_ptr);
    (
        nup,
        partition_slice,
        partition,
        log_posterior_predictive,
        rng,
    )
}

unsafe fn neal_algorithm3_push_into_slice(partition: &Partition, slice: &mut [i32]) {
    partition.labels_into_slice(slice, |x| x.unwrap() as i32);
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__neal_algorithm3_crp(
    n_updates_for_partition: i32,
    n_items: i32,
    partition_ptr: *mut i32,
    prior_only: i32,
    log_posterior_predictive_function_ptr: *const c_void,
    env_ptr: *const c_void,
    seed_ptr: *const i32, // Assumed length is 32
    mass: f64,
) -> () {
    let (nup, partition_slice, partition, log_posterior_predictive, mut rng) =
        neal_algorithm3_process_arguments(
            n_updates_for_partition,
            n_items,
            partition_ptr,
            prior_only,
            log_posterior_predictive_function_ptr,
            env_ptr,
            seed_ptr,
        );
    let neal_functions = crp::CRPParameters::new(Mass::new(mass));
    let partition = update_neal_algorithm3(
        nup,
        &partition,
        &neal_functions,
        &log_posterior_predictive,
        &mut rng,
    );
    neal_algorithm3_push_into_slice(&partition, partition_slice);
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__neal_algorithm3_nggp(
    n_updates_for_partition: i32,
    n_items: i32,
    partition_ptr: *mut i32,
    prior_only: i32,
    log_posterior_predictive_function_ptr: *const c_void,
    env_ptr: *const c_void,
    seed_ptr: *const i32, // Assumed length is 32
    n_updates_for_u: i32,
    u_ptr: *mut f64,
    mass: f64,
    reinforcement: f64,
) -> () {
    let (nup, partition_slice, partition, log_posterior_predictive, mut rng) =
        neal_algorithm3_process_arguments(
            n_updates_for_partition,
            n_items,
            partition_ptr,
            prior_only,
            log_posterior_predictive_function_ptr,
            env_ptr,
            seed_ptr,
        );
    let nuu = n_updates_for_u as u32;
    let u = UinNGGP::new(*u_ptr);
    let mass = Mass::new(mass);
    let reinforcement = Reinforcement::new(reinforcement);
    let neal_functions = nggp::NGGPParameters::new(u, mass, reinforcement);
    let partition = update_neal_algorithm3(
        nup,
        &partition,
        &neal_functions,
        &log_posterior_predictive,
        &mut rng,
    );
    *u_ptr = super::nggp::update_u(u, &partition, mass, reinforcement, nuu, &mut rng).unwrap();
    neal_algorithm3_push_into_slice(&partition, partition_slice);
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__neal_algorithm3_frp(
    n_updates_for_partition: i32,
    n_items: i32,
    partition_ptr: *mut i32,
    prior_only: i32,
    log_posterior_predictive_function_ptr: *const c_void,
    env_ptr: *const c_void,
    seed_ptr: *const i32, // Assumed length is 32
    focal_ptr: *const i32,
    weights_ptr: *const f64,
    permutation_ptr: *const i32,
    mass: f64,
) -> () {
    let (nup, partition_slice, partition, log_posterior_predictive, mut rng) =
        neal_algorithm3_process_arguments(
            n_updates_for_partition,
            n_items,
            partition_ptr,
            prior_only,
            log_posterior_predictive_function_ptr,
            env_ptr,
            seed_ptr,
        );
    let focal_slice = slice::from_raw_parts(focal_ptr, partition.n_items());
    let focal = Partition::from(focal_slice);
    let weights_slice = slice::from_raw_parts(weights_ptr, focal.n_subsets());
    let weights = Weights::from(weights_slice).unwrap();
    let permutation_slice = slice::from_raw_parts(permutation_ptr, focal.n_items());
    let permutation_vector: Vec<usize> = permutation_slice.iter().map(|x| *x as usize).collect();
    let permutation = Permutation::from_vector(permutation_vector).unwrap();
    let mass = Mass::new(mass);
    let neal_functions = frp::FRPParameters::new(&focal, &weights, &permutation, mass).unwrap();
    let partition = update_neal_algorithm3_generalized(
        nup,
        &partition,
        &permutation,
        &neal_functions,
        &log_posterior_predictive,
        &mut rng,
    );
    neal_algorithm3_push_into_slice(&partition, partition_slice);
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__focalrw_crp(
    n_attempts: i32,
    n_items: i32,
    partition_ptr: *mut i32,
    rate: f64,
    mass: f64,
    prior_only: i32,
    log_likelihood_function_ptr: *const c_void,
    env_ptr: *const c_void,
    seed_ptr: *const i32, // Assumed length is 32
    n_accepts: *mut i32,
    crp_mass: f64,
) -> () {
    let na = n_attempts as u32;
    let ni = n_items as usize;
    let partition_slice = slice::from_raw_parts_mut(partition_ptr, ni);
    let partition = Partition::from(partition_slice);
    let parameters = CRPParameters::new(Mass::new(crp_mass));
    let log_prior = |p: &Partition| crate::crp::log_pmf(&p, &parameters);
    let log_likelihood: Box<dyn Fn(&[usize]) -> f64> = if prior_only != 0 {
        Box::new(|_indices: &[usize]| 0.0)
    } else {
        Box::new(move |indices: &[usize]| {
            callRFunction_logIntegratedLikelihoodOfSubset(
                log_likelihood_function_ptr,
                RR_SEXP_vector_INTSXP::from_slice(indices),
                env_ptr,
            )
        })
    };
    let log_target = make_posterior(log_prior, log_likelihood);
    let rate = Rate::new(rate);
    let mass = Mass::new(mass);
    let mut rng = mk_rng_isaac(seed_ptr);
    let results = update_rwmh(na, &partition, rate, mass, &log_target, &mut rng);
    results
        .0
        .labels_into_slice(partition_slice, |x| x.unwrap() as i32);
    *n_accepts = results.1 as i32;
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__focalrw_frp(
    n_attempts: i32,
    n_items: i32,
    partition_ptr: *mut i32,
    rate: f64,
    mass: f64,
    prior_only: i32,
    log_likelihood_function_ptr: *const c_void,
    env_ptr: *const c_void,
    seed_ptr: *const i32, // Assumed length is 32
    n_accepts: *mut i32,
    frp_partition_ptr: *const i32,
    frp_weights_ptr: *const f64,
    frp_permutation_ptr: *const i32,
    frp_mass: f64,
) -> () {
    let na = n_attempts as u32;
    let ni = n_items as usize;
    let partition_slice = slice::from_raw_parts_mut(partition_ptr, ni);
    let partition = Partition::from(partition_slice);
    let frp_partition_slice = slice::from_raw_parts(frp_partition_ptr, ni);
    let frp_partition = Partition::from(frp_partition_slice);
    let frp_weights_slice = slice::from_raw_parts(frp_weights_ptr, frp_partition.n_subsets());
    let frp_weights = Weights::from(frp_weights_slice).unwrap();
    let frp_permutation_slice = slice::from_raw_parts(frp_permutation_ptr, ni);
    let mut frp_permutation_slice2 = Vec::with_capacity(ni);
    for i in 0..ni {
        frp_permutation_slice2.push(frp_permutation_slice[i] as usize);
    }
    let frp_permutation = Permutation::from_vector(frp_permutation_slice2).unwrap();
    let parameters = FRPParameters::new(
        &frp_partition,
        &frp_weights,
        &frp_permutation,
        Mass::new(frp_mass),
    )
    .unwrap();
    let log_prior = |p: &Partition| crate::frp::log_pmf(&p, &parameters);
    let log_likelihood: Box<dyn Fn(&[usize]) -> f64> = if prior_only != 0 {
        Box::new(|_indices: &[usize]| 0.0)
    } else {
        Box::new(move |indices: &[usize]| {
            callRFunction_logIntegratedLikelihoodOfSubset(
                log_likelihood_function_ptr,
                RR_SEXP_vector_INTSXP::from_slice(indices),
                env_ptr,
            )
        })
    };
    let log_target = make_posterior(log_prior, log_likelihood);
    let rate = Rate::new(rate);
    let mass = Mass::new(mass);
    let mut rng = mk_rng_isaac(seed_ptr);
    let results = update_rwmh(na, &partition, rate, mass, &log_target, &mut rng);
    results
        .0
        .labels_into_slice(partition_slice, |x| x.unwrap() as i32);
    *n_accepts = results.1 as i32;
}
