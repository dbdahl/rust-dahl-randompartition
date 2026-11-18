//! Partition updates preserve their seeded behavior on success. On error,
//! they stop at the first failing callback, retain completed visits, and
//! restore the failed visit's allocation with valid cluster bookkeeping.
//! Each error case fails every call of a successful run in turn.

use dahl_randompartition::clust::Clustering;
use dahl_randompartition::crp::CrpParameters;
use dahl_randompartition::distr::FullConditional;
use dahl_randompartition::mcmc::{
    try_update_neal_algorithm_full, try_update_neal_algorithm3, try_update_neal_algorithm8,
    try_update_partition_gibbs, update_neal_algorithm_full, update_neal_algorithm3,
    update_neal_algorithm8, update_partition_gibbs,
};
use dahl_randompartition::perm::Permutation;
use dahl_randompartition::prelude::*;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use std::cell::RefCell;

const DATA: [f64; 6] = [-1.0, -0.9, 0.1, 0.2, 1.0, 1.1];
const N_UPDATES: u32 = 2;

// A prior that records the partition at the start of each visit to an item,
// when the update asks for the item's full conditional.
struct Recording<'a, P> {
    prior: &'a P,
    starts: RefCell<Vec<Vec<usize>>>,
}

impl<P: FullConditional> FullConditional for Recording<'_, P> {
    fn log_full_conditional(&self, item: usize, clustering: &Clustering) -> Vec<(usize, f64)> {
        self.starts
            .borrow_mut()
            .push(clustering.allocation().clone());
        self.prior.log_full_conditional(item, clustering)
    }
}

#[derive(Clone, Copy, Debug)]
enum Update {
    Gibbs,
    Algorithm3,
    Algorithm8,
    Full,
}

struct Run {
    result: Result<(), usize>,
    allocation: Vec<usize>,
    calls: usize,
    // The visit, an index into 'starts', during which each call was made.
    call_visits: Vec<usize>,
    starts: Vec<Vec<usize>>,
    next_random: u64,
}

fn assert_valid_clustering(state: &Clustering) {
    let expected = Clustering::from_vector(state.allocation().clone());
    assert_eq!(state.n_items_allocated(), state.n_items());
    assert_eq!(state.n_clusters(), expected.n_clusters());
    let mut active_labels = state.active_labels().clone();
    active_labels.sort_unstable();
    assert_eq!(&active_labels, expected.active_labels());
    for label in 0..=state.max_label().max(expected.max_label()) + 1 {
        assert_eq!(
            state.size_of(label),
            expected.size_of(label),
            "label {label}"
        );
    }
    assert!(!state.allocation().contains(&state.new_label()));
}

// The sum of squares within the clusters of 'allocation', so that the draws
// depend on the data.
fn within_sum_of_squares(allocation: &[usize]) -> f64 {
    let mut total = 0.0;
    for label in 0..=allocation.iter().copied().max().unwrap_or(0) {
        let x: Vec<f64> = (0..allocation.len())
            .filter(|&i| allocation[i] == label)
            .map(|i| DATA[i])
            .collect();
        if !x.is_empty() {
            let mean = x.iter().sum::<f64>() / x.len() as f64;
            total += x.iter().map(|y| (y - mean).powi(2)).sum::<f64>();
        }
    }
    total
}

fn log_posterior_predictive(item: usize, items: &[usize]) -> f64 {
    let mean = items.iter().map(|&i| DATA[i]).sum::<f64>() / (items.len() + 1) as f64;
    -(DATA[item] - mean).powi(2)
}

fn log_likelihood_contribution(item: usize, scale: &f64, label: usize, is_new: bool) -> f64 {
    let center = if is_new { 0.0 } else { label as f64 - 1.0 };
    -scale * (DATA[item] - center).powi(2)
}

fn common_item_cache(item: usize) -> f64 {
    1.0 + item as f64 / 10.0
}

// A run of 'update' whose closure fails at call 'fail_at', counting from 1, or
// never for 0.
fn run(update: Update, fail_at: usize) -> Run {
    let n_items = DATA.len();
    let prior = CrpParameters::new(n_items, Concentration::new(1.0).unwrap());
    let recording = Recording {
        prior: &prior,
        starts: RefCell::new(Vec::new()),
    };
    let permutation = Permutation::from_vector(vec![3, 0, 5, 1, 4, 2]).unwrap();
    let mut state = Clustering::from_vector(vec![0, 0, 1, 1, 2, 0]);
    let mut rng = Pcg64Mcg::seed_from_u64(7);
    let mut calls = 0;
    let mut call_visits = Vec::new();
    let mut call = |value: f64| {
        calls += 1;
        call_visits.push(recording.starts.borrow().len() - 1);
        if calls == fail_at {
            Err(calls)
        } else {
            Ok(value)
        }
    };
    let result = match update {
        Update::Gibbs => try_update_partition_gibbs(
            N_UPDATES,
            &mut state,
            &permutation,
            &recording,
            &mut |_item: usize, state: &Clustering| {
                call(-within_sum_of_squares(state.allocation()))
            },
            &mut rng,
        ),
        Update::Algorithm3 => try_update_neal_algorithm3(
            N_UPDATES,
            &mut state,
            &permutation,
            &recording,
            &mut |item: usize, items: &[usize]| call(log_posterior_predictive(item, items)),
            &mut rng,
        ),
        Update::Algorithm8 => try_update_neal_algorithm8(
            N_UPDATES,
            &mut state,
            &permutation,
            &recording,
            &mut |item: usize, scale: &f64, label: usize, is_new: bool| {
                call(log_likelihood_contribution(item, scale, label, is_new))
            },
            common_item_cache,
            &mut rng,
        ),
        Update::Full => try_update_neal_algorithm_full(
            N_UPDATES,
            &mut state,
            &permutation,
            &recording,
            &mut |state: &Clustering| call(-within_sum_of_squares(state.allocation())),
            &mut rng,
        ),
    };
    assert_valid_clustering(&state);
    // A caller can keep using the restored partition, including allocating
    // new clusters after a failed visit.
    let mut probe = state.clone();
    for item in 0..n_items {
        probe.allocate(item, probe.new_label());
        assert_valid_clustering(&probe);
        probe.allocate(item, state.get(item));
        assert_eq!(probe.allocation(), state.allocation());
        assert_valid_clustering(&probe);
    }
    Run {
        result,
        allocation: state.allocation().clone(),
        calls,
        call_visits,
        starts: recording.starts.into_inner(),
        next_random: rng.next_u64(),
    }
}

fn check(update: Update) {
    let success = run(update, 0);
    assert_eq!(success.result, Ok(()));
    assert!(success.calls > 0);
    for k in 1..=success.calls {
        let failure = run(update, k);
        assert_eq!(failure.result, Err(k), "{update:?}, call {k}");
        assert_eq!(
            failure.calls, k,
            "{update:?} called its closure after call {k} failed"
        );
        // The runs agree until the failure, so the state is the one at the
        // start of the visit during which call k was made.
        let visit = success.call_visits[k - 1];
        assert_eq!(
            failure.allocation, success.starts[visit],
            "{update:?}, call {k}, during visit {visit}"
        );
    }
}

#[test]
fn gibbs_stops_at_the_first_error() {
    check(Update::Gibbs);
}

#[test]
fn neal_algorithm3_stops_at_the_first_error() {
    check(Update::Algorithm3);
}

#[test]
fn neal_algorithm8_stops_at_the_first_error() {
    check(Update::Algorithm8);
}

#[test]
fn neal_algorithm_full_stops_at_the_first_error() {
    check(Update::Full);
}

fn run_infallible(update: Update) -> (Clustering, u64) {
    let n_items = DATA.len();
    let prior = CrpParameters::new(n_items, Concentration::new(1.0).unwrap());
    let permutation = Permutation::from_vector(vec![3, 0, 5, 1, 4, 2]).unwrap();
    let mut state = Clustering::from_vector(vec![0, 0, 1, 1, 2, 0]);
    let mut rng = Pcg64Mcg::seed_from_u64(7);
    match update {
        Update::Gibbs => update_partition_gibbs(
            N_UPDATES,
            &mut state,
            &permutation,
            &prior,
            &mut |_: usize, state: &Clustering| -within_sum_of_squares(state.allocation()),
            &mut rng,
        ),
        Update::Algorithm3 => update_neal_algorithm3(
            N_UPDATES,
            &mut state,
            &permutation,
            &prior,
            &log_posterior_predictive,
            &mut rng,
        ),
        Update::Algorithm8 => update_neal_algorithm8(
            N_UPDATES,
            &mut state,
            &permutation,
            &prior,
            &mut log_likelihood_contribution,
            common_item_cache,
            &mut rng,
        ),
        Update::Full => update_neal_algorithm_full(
            N_UPDATES,
            &mut state,
            &permutation,
            &prior,
            &mut |state: &Clustering| -within_sum_of_squares(state.allocation()),
            &mut rng,
        ),
    }
    assert_valid_clustering(&state);
    (state, rng.next_u64())
}

#[test]
fn successful_updates_preserve_seeded_behavior() {
    // Reference results from the original infallible implementations, before
    // they delegated to the try_ forms. Pin the RNG position as well as the
    // labels so changes to random-number consumption are detected.
    let expected_next_random = 13_501_250_456_227_795_994;
    for (update, allocation, calls) in [
        (Update::Gibbs, [2, 2, 2, 2, 0, 1], 38),
        (Update::Algorithm3, [2, 2, 1, 2, 0, 1], 38),
        (Update::Algorithm8, [0, 0, 1, 1, 4, 2], 57),
        (Update::Full, [2, 2, 2, 2, 0, 1], 38),
    ] {
        let fallible = run(update, 0);
        assert_eq!(fallible.result, Ok(()), "{update:?}");
        assert_eq!(fallible.allocation, allocation, "{update:?}");
        assert_eq!(fallible.calls, calls, "{update:?}");
        assert_eq!(fallible.starts.len(), N_UPDATES as usize * DATA.len());
        assert_eq!(fallible.next_random, expected_next_random, "{update:?}");

        let (infallible, next_random) = run_infallible(update);
        assert_eq!(infallible.allocation(), &allocation, "{update:?}");
        assert_eq!(next_random, expected_next_random, "{update:?}");
    }
}
