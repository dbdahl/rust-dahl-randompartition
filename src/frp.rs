extern crate rand;
extern crate dahl_partition;
extern crate dahl_salso;

use dahl_partition::*;
use rand::seq::SliceRandom;
use rand::distributions::{Distribution, WeightedIndex};
use std::convert::TryFrom;
use std::slice;

fn ensure_empty_subset(partition: &mut Partition) {
    match partition.subsets().last() {
        None => partition.new_subset(),
        Some(last) => {
            if !last.is_empty() {
                partition.new_subset()
            }
        }
    }
}

pub fn sample(focal: &Partition, weights: &[f64], mass: f64) -> Partition {
    assert!(focal.is_canonical());
    assert_eq!(focal.n_subsets(), weights.len());
    assert!(weights.iter().all(|w| !w.is_nan() && w.is_finite() && *w >= 0.0));
    assert!(mass > 0.0, "Mass must be greater than 0.0.");
    let ni = focal.n_items();
    let mut rng = rand::thread_rng();
    let mut p = Partition::new(ni);
    let mut permutation: Vec<usize> = (0..ni).collect();
    permutation.shuffle(&mut rng);
    let mut permutation_subset = Subset::new();
    for i in 0..ni {
        let ii = permutation[i];
        permutation_subset.add(ii);
        let focal_subset_index = focal.label_of(ii).unwrap();
        let focal_subset = &focal.subsets()[focal_subset_index];
        let this_weight = weights[focal_subset_index];
        let focal_permutation_intersection= focal_subset.intersection(&permutation_subset);
        let denominator= focal_permutation_intersection.n_items() as f64;
        ensure_empty_subset(&mut p);
        let probs = p.subsets().iter().map(|subset| {
            if subset.is_empty() {
                mass + if focal_permutation_intersection.n_items() == 0 && focal_permutation_intersection.contains(ii) { this_weight } else { 0.0 }
            } else {
                ( subset.n_items() as f64 ) + this_weight * (focal_subset.intersection(&subset).n_items() as f64) / denominator
            }
        });
        let dist = WeightedIndex::new(probs).unwrap();
        let subset_index = dist.sample(&mut rng);
        p.add_with_index(i, subset_index);
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample() {
        let n_partitions = 10000;
        let n_items = 4;
        let mass = 2.0;
        let mut samples = PartitionsHolder::with_capacity(n_partitions, n_items);
        let focal = Partition::one_subset(n_items);
        let weight = &[0.0];
        for _ in 0..n_partitions {
            samples.push_partition(&sample(&focal, weight, mass));
        }
        let mut psm = dahl_salso::psm::psm(&samples.view(), true);
        let truth = 1.0 / (1.0 + mass);
        let margin_of_error = 3.58 * (truth * (1.0 - truth) / n_partitions as f64).sqrt();
        assert!(psm.view().data().iter().all(|prob| {
            *prob == 1.0 || (truth - margin_of_error < *prob && *prob < truth + margin_of_error)
        }));
    }

}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__rfp__sample(
    n_partitions: i32,
    n_items: i32,
    focal_ptr: *const i32,
    weights_ptr: *const f64,
    mass: f64,
    ptr: *mut i32,
) -> () {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let array: &mut [i32] = slice::from_raw_parts_mut(ptr, np * ni);
    let focal = Partition::from(slice::from_raw_parts(focal_ptr, ni));
    let weights = slice::from_raw_parts(weights_ptr, focal.n_subsets());
    for i in 0..np {
        let p = sample(&focal, weights, mass);
        let labels = p.labels();
        for j in 0..ni {
            array[np * j + i] = i32::try_from(labels[j].unwrap()).unwrap();
        }
    }
}
