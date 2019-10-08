use crate::frp::{engine, Weights};
use crate::prelude::*;
use dahl_partition::*;
use rand::thread_rng;
use rand::Rng;

fn update(
    n_updates: u32,
    current: &mut Partition,
    rate: Rate,
    mass: Mass,
    log_target: fn(&Partition) -> f64,
) -> (Partition, u32) {
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
