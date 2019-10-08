/*
use crate::prelude::*;
use dahl_partition::*;
use rand::thread_rng;

fn mhrw(state: Partition, rate: Rate, mass: Mass, target: fn(Partition) -> f64) -> Partition {
    let mut rng = thread_rng();
    let permutation = Permutation::random(state.n_items(), &mut rng);
    Partition::one_subset(state.n_items())
}
*/
