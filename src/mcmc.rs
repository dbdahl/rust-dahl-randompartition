use crate::frp;
use crate::prelude::*;
use dahl_partition::*;
use dahl_roxido::mk_rng_isaac;
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use std::ffi::c_void;
use std::slice;

fn update_neal_algorithm3<T, U, V>(
    n_updates: u32,
    current: &Partition,
    weight_for_new_subset: f64,
    weight_for_existing_subset: &U,
    log_posterior_predictive: &T,
    rng: &mut V,
) -> Partition
where
    T: Fn(usize, &[usize]) -> f64,
    U: Fn(usize) -> f64,
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
            let weights = state.subsets().iter().map(|subset| {
                let pp = if subset.is_empty() {
                    weight_for_new_subset
                } else {
                    weight_for_existing_subset(subset.n_items())
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
    if empty_subset_at_end {
        state.pop_subset();
    } else {
        state.canonicalize();
    }
    state
}

/*
fn update_rwmh<T, U>(
    n_attempts: u32,
    current: &Partition,
    rate: NonnegativeDouble,
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
    let mut weights_state = frp::Weights::constant(rate.as_f64(), state.n_subsets());
    let rng_wrapper = TargetOrRandom::Random(rng);
    for _ in 0..n_attempts {
        state.canonicalize();
        permutation.shuffle(rng_wrapper.get_rng());
        let proposal = frp::engine(&state, &weights_state, &permutation, mass, rng_wrapper);
        let weights_proposal = frp::Weights::constant(rate.as_f64(), proposal.0.n_subsets());
        let log_target_proposal = log_target(&proposal.0);
        let log_ratio_target = log_target_proposal - log_target_state;
        let log_ratio_proposal = frp::engine::<U>(
            &proposal.0,
            &weights_proposal,
            &permutation,
            mass,
            TargetOrRandom::Target(&mut state),
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
    (state, accepts)
}
*/

fn make_posterior<'a, T: 'a, U: 'a>(
    log_prior: T,
    log_likelihood: U,
) -> Box<dyn Fn(&Partition) -> f64 + 'a>
where
    T: Fn(&Partition) -> f64,
    U: Fn(&[usize]) -> f64,
{
    let log_target = move |partition: &Partition| {
        partition
            .subsets()
            .iter()
            .fold(log_prior(partition), |sum, subset| {
                sum + log_likelihood(&subset.items()[..])
            })
    };
    Box::new(log_target)
}

#[cfg(test)]
mod tests_mcmc {
    use super::*;
    use rand::thread_rng;

    /*
    #[test]
    fn test_crp_rwmh() {
        let n_items = 5;
        let mut current = Partition::one_subset(n_items);
        let rate = NonnegativeDouble::new(5.0);
        let mass = Mass::new(1.0);
        let log_prior = |p: &Partition| crate::crp::log_pmf(&p, mass);
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
    */

    #[test]
    fn test_crp_neal_algorithm3() {
        let n_items = 5;
        let mut current = Partition::one_subset(n_items);
        let mass = Mass::new(1.0);
        let log_posterior_predictive = |_i: usize, _indices: &[usize]| 0.0;
        let weight_for_existing_subset = |size: usize| size as f64;
        let mut sum = 0;
        let n_samples = 10000;
        for _ in 0..n_samples {
            current = update_neal_algorithm3(
                2,
                &current,
                mass.as_f64(),
                &weight_for_existing_subset,
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
    fn callRFunction_logIntegratedLikelihoodOfItem(
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

pub enum PartitionPrior {
    CRP {
        mass: Mass,
    },
    Focal {
        focal: Partition,
        permutation: Permutation,
        weights: frp::Weights,
        mass: Mass,
    },
}

const PRIOR_PARTITION_CODE_CRP: i32 = 0;
const PRIOR_PARTITION_CODE_NGGP: i32 = 1;
const PRIOR_PARTITION_CODE_EPA: i32 = 2;
const PRIOR_PARTITION_CODE_FOCAL: i32 = 3;

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__neal_algorithm3_update(
    n_updates_for_partition: i32,
    n_updates_for_u: i32,
    n_items: i32,
    prior_partition_code: i32,
    u: *mut f64,
    mass: f64,
    reinforcement: f64,
    partition_ptr: *mut i32,
    prior_only: i32,
    log_likelihood_function_ptr: *const c_void,
    env_ptr: *const c_void,
    seed_ptr: *const i32, // Assumed length is 32
) -> () {
    let nup = n_updates_for_partition as u32;
    let nuu = n_updates_for_u as u32;
    let ni = n_items as usize;
    let partition_slice = slice::from_raw_parts_mut(partition_ptr, ni);
    let partition = Partition::from(partition_slice);
    let mass = Mass::new(mass);
    let reinforcement = Reinforcement::new(reinforcement);
    let log_posterior_predictive: Box<dyn Fn(usize, &[usize]) -> f64> = if prior_only != 0 {
        Box::new(|_i: usize, _indices: &[usize]| 0.0)
    } else {
        Box::new(|i: usize, indices: &[usize]| {
            callRFunction_logIntegratedLikelihoodOfItem(
                log_likelihood_function_ptr,
                (i as i32) + 1,
                RR_SEXP_vector_INTSXP::from_slice(indices),
                env_ptr,
            )
        })
    };
    let (weight_for_new_subset, weight_for_existing_subset): (f64, Box<dyn Fn(usize) -> f64>) =
        match prior_partition_code {
            PRIOR_PARTITION_CODE_CRP => (mass.as_f64(), Box::new(|size| size as f64)),
            PRIOR_PARTITION_CODE_NGGP => (
                mass.as_f64() * (*u + 1.0).powf(reinforcement.as_f64()),
                Box::new(|size| size as f64 - reinforcement.as_f64()),
            ),
            _ => panic!("Unsupported prior partition code."),
        };
    let mut rng = mk_rng_isaac(seed_ptr);
    let partition = update_neal_algorithm3(
        nup,
        &partition,
        weight_for_new_subset,
        &weight_for_existing_subset,
        &log_posterior_predictive,
        &mut rng,
    );
    if prior_partition_code == PRIOR_PARTITION_CODE_NGGP {
        *u = super::nggp::update_u(
            NonnegativeDouble::new(*u),
            &partition,
            mass,
            reinforcement,
            nuu,
        )
        .as_f64();
    };
    partition.labels_into_slice(partition_slice, |x| x.unwrap() as i32);
}

/*
#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__mhrw_update(
    n_attempts: i32,
    n_items: i32,
    rate: f64,
    mass: f64,
    partition_ptr: *mut i32,
    log_likelihood_function_ptr: *const c_void,
    env_ptr: *const c_void,
    n_accepts: *mut i32,
    seed_ptr: *const i32, // Assumed length is 32
) -> () {
    let na = n_attempts as u32;
    let ni = n_items as usize;
    let partition_slice = slice::from_raw_parts_mut(partition_ptr, ni);
    let partition = Partition::from(partition_slice);
    let rate = NonnegativeDouble::new(rate);
    let mass = Mass::new(mass);
    let log_prior = |p: &Partition| crate::crp::log_pmf(&p, mass);
    let log_likelihood = |indices: &[usize]| {
        callRFunction_logIntegratedLikelihoodOfSubset(
            log_likelihood_function_ptr,
            RR_SEXP_vector_INTSXP::from_slice(indices),
            env_ptr,
        )
    };
    let log_target = make_posterior(log_prior, log_likelihood);
    let mut rng = mk_rng_isaac(seed_ptr);
    let results = update_rwmh(na, &partition, rate, mass, &log_target, &mut rng);
    results
        .0
        .labels_into_slice(partition_slice, |x| x.unwrap() as i32);
    *n_accepts = results.1 as i32;
}
*/
