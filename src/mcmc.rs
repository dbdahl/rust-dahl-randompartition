use crate::frp::{engine, Weights};
use crate::prelude::*;
use dahl_partition::*;
use rand::thread_rng;
use rand::Rng;
use std::ffi::c_void;
use std::slice;

fn update<T>(
    n_updates: u32,
    current: &mut Partition,
    rate: Rate,
    mass: Mass,
    log_target: T,
) -> (Partition, u32)
where
    T: Fn(&Partition) -> f64,
{
    let mut accepts: u32 = 0;
    let mut state = current.clone();
    let mut rng = thread_rng();
    let mut permutation = Permutation::natural(state.n_items());
    let mut log_target_state = log_target(&state);
    let mut weights_state = Weights::constant(rate.as_f64(), state.n_subsets());
    for _ in 0..n_updates {
        permutation.shuffle(&mut rng);
        let proposal = engine(&state, &weights_state, &permutation, mass, None);
        let weights_proposal = Weights::constant(rate.as_f64(), proposal.0.n_subsets());
        let log_target_proposal = log_target(&proposal.0);
        let log_ratio_target = log_target_proposal - log_target_state;
        let log_ratio_proposal = engine(
            &proposal.0,
            &weights_proposal,
            &permutation,
            mass,
            Some(&mut state),
        )
        .1 - proposal.1;
        let log_mh_ratio = log_ratio_target + log_ratio_proposal;
        if log_mh_ratio >= 1.0 || rng.gen_range(0.0, 1.0) < log_mh_ratio.exp() {
            accepts += 1;
            state = proposal.0;
            log_target_state = log_ratio_proposal;
            weights_state = weights_proposal;
        };
    }
    (state, accepts)
}

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

fn update_under_posterior<T, U>(
    n_updates: u32,
    current: &mut Partition,
    rate: Rate,
    mass: Mass,
    log_prior: T,
    log_likelihood: U,
) -> (Partition, u32)
where
    T: Fn(&Partition) -> f64,
    U: Fn(&[usize]) -> f64,
{
    let log_target = |partition: &Partition| {
        partition
            .subsets()
            .iter()
            .fold(log_prior(partition), |sum, subset| {
                sum + log_likelihood(&subset.items()[..])
            })
    };
    update(n_updates, current, rate, mass, log_target)
}

#[cfg(test)]
mod tests_mcmc {
    use super::*;

    #[test]
    fn test_crp() {
        let n_items = 5;
        let mut current = Partition::one_subset(n_items);
        let rate = Rate::new(5.0);
        let mass = Mass::new(1.0);
        let log_prior = |p: &Partition| crate::crp::pmf(&p, mass);
        let log_likelihood = |_indices: &[usize]| 0.0;
        let mut sum = 0;
        let n_samples = 10000;
        for _ in 0..n_samples {
            let result =
                update_under_posterior(1, &mut current, rate, mass, log_prior, log_likelihood);
            current = result.0;
            sum += current.n_subsets();
        }
        let mean_number_of_subsets = (sum as f64) / (n_samples as f64);
        let z_stat = (mean_number_of_subsets - 2.283333) / (0.8197222 / n_samples as f64).sqrt();
        assert!(z_stat.abs() < 3.290527);
    }
}

extern "C" {
    fn callRFunction_indices_f64(
        fn_ptr: *const c_void,
        indices_ptr: *const c_void,
        len: i32,
        env_ptr: *const c_void,
    ) -> f64;
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__mhrw_update(
    n_updates: i32,
    n_items: i32,
    rate: f64,
    mass: f64,
    partition_ptr: *mut i32,
    log_likelihood_function_ptr: *const c_void,
    env_ptr: *const c_void,
    n_accepts: *mut i32,
) -> () {
    let nu = n_updates as u32;
    let ni = n_items as usize;
    let partition_slice = slice::from_raw_parts_mut(partition_ptr, ni);
    let mut partition = Partition::from(partition_slice);
    let rate = Rate::new(rate);
    let mass = Mass::new(mass);
    let log_prior = |p: &Partition| crate::crp::pmf(&p, mass);
    let log_likelihood = |indices: &[usize]| {
        let indices_ptr = indices.as_ptr() as *const c_void;
        callRFunction_indices_f64(
            log_likelihood_function_ptr,
            indices_ptr,
            indices.len() as i32,
            env_ptr,
        )
    };
    let log_target = make_posterior(log_prior, log_likelihood);
    let results = update(nu, &mut partition, rate, mass, log_target);
    results
        .0
        .labels_into_slice(partition_slice, |x| x.unwrap() as i32);
    *n_accepts = results.1 as i32;
}
