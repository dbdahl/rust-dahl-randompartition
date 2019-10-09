use crate::frp::{engine, Weights};
use crate::prelude::*;
use dahl_partition::*;
use rand::thread_rng;
use rand::Rng;

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

#[cfg(test)]
mod tests_mcmc {
    use super::*;

    #[test]
    fn test_crp() {
        let n_items = 5;
        let mut current = Partition::one_subset(n_items);
        let mut n_accepts = 0;
        let rate = Rate::new(5.0);
        let mass = Mass::new(1.0);
        let log_target = |p: &Partition| crate::crp::pmf(&p, mass);
        let mut sum = 0;
        let n_samples = 10000;
        for _ in 0..n_samples {
            let result = update(1, &mut current, rate, mass, log_target);
            current = result.0;
            n_accepts += result.1;
            sum += current.n_subsets();
        }
        let mean_number_of_subsets = (sum as f64) / (n_samples as f64);
        let Zstat = (mean_number_of_subsets - 2.283333) / (0.8197222 / n_samples as f64).sqrt();
        assert!(Zstat.abs() < 3.290527);
    }
}
