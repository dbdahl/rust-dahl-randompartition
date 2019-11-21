// Ewens Pitman attraction partition distribution

use crate::prelude::*;
use crate::TargetOrRandom;

use dahl_partition::*;
use dahl_roxido::mk_rng_isaac;
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::*;
use rand_isaac::IsaacRng;
use std::convert::TryFrom;
use std::slice;

pub struct DistanceBorrower<'a>(SquareMatrixBorrower<'a>);
pub struct SimilarityBorrower<'a>(SquareMatrixBorrower<'a>);

pub struct EPAParameters<'a, 'b> {
    similarity: &'a SimilarityBorrower<'a>,
    permutation: &'b Permutation,
    mass: Mass,
    discount: Discount,
}

impl<'a, 'b> EPAParameters<'a, 'b> {
    pub fn new(
        similarity: &'a SimilarityBorrower,
        permutation: &'b Permutation,
        mass: Mass,
        discount: Discount,
    ) -> Self {
        Self {
            similarity,
            permutation,
            mass,
            discount,
        }
    }
}

pub fn engine<T: Rng>(
    parameters: &EPAParameters,
    mut target_or_rng: TargetOrRandom<T>,
) -> (Partition, f64) {
    let ni = parameters.similarity.0.n_items();
    assert_eq!(
        ni,
        parameters.permutation.len(),
        "Number of items in similarity and permutation are not the same."
    );
    let mass = parameters.mass.unwrap();
    if let TargetOrRandom::Target(t) = &mut target_or_rng {
        assert_eq!(
            t.n_items(),
            ni,
            "Number of items in target and similarity are not the same."
        );
        t.canonicalize_by_permutation(Some(&parameters.permutation));
    };
    let mut log_probability = 0.0;
    let mut partition = Partition::new(ni);
    for i in 0..ni {
        let ii = parameters.permutation[i];
        // Ensure there is an empty subset
        match partition.subsets().last() {
            None => partition.new_subset(),
            Some(last) => {
                if !last.is_empty() {
                    partition.new_subset()
                }
            }
        }
        let qt = (partition.n_subsets() - 1) as f64; // Since one subset is empty.
        let kt = (i as f64) - parameters.discount * qt;
        let denominator = parameters
            .similarity
            .0
            .sum_of_row_subset(ii, parameters.permutation.slice_until(i));
        let probs: Vec<(usize, f64)> = partition
            .subsets()
            .iter()
            .enumerate()
            .map(|(subset_index, subset)| {
                let prob = if subset.is_empty() {
                    mass + parameters.discount * qt
                } else {
                    kt * parameters
                        .similarity
                        .0
                        .sum_of_row_subset(ii, &subset.items()[..])
                        / denominator
                };
                (subset_index, prob)
            })
            .collect();
        let subset_index = match &mut target_or_rng {
            TargetOrRandom::Random(rng) => {
                let dist = WeightedIndex::new(probs.iter().map(|x| x.1)).unwrap();
                dist.sample(*rng)
            }
            TargetOrRandom::Target(t) => t.label_of(ii).unwrap(),
        };
        let numerator = probs[subset_index].1;
        let denominator = probs.iter().fold(0.0, |sum, x| sum + x.1);
        log_probability += (numerator / denominator).ln();
        partition.add_with_index(ii, subset_index);
    }
    partition.canonicalize();
    (partition, log_probability)
}

pub fn sample<T: Rng>(parameters: &EPAParameters, rng: &mut T) -> Partition {
    engine(&parameters, TargetOrRandom::Random(rng)).0
}

pub fn log_pmf(target: &mut Partition, parameters: &EPAParameters) -> f64 {
    engine(&parameters, TargetOrRandom::Target::<IsaacRng>(target)).1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample() {
        let n_partitions = 10000;
        let n_items = 4;
        let mass = Mass::new(2.0);
        let discount = Discount::new(0.0);
        let mut samples = PartitionsHolder::with_capacity(n_partitions, n_items);
        let mut ones = SquareMatrix::ones(n_items);
        let similarity = SimilarityBorrower(ones.view());
        let rng = &mut thread_rng();
        let permutation = Permutation::random(n_items, rng);
        let parameters = EPAParameters::new(&similarity, &permutation, mass, discount);
        for _ in 0..n_partitions {
            samples.push_partition(&sample(&parameters, rng));
        }
        let mut psm = dahl_salso::psm::psm(&samples.view(), true);
        let truth = 1.0 / (1.0 + mass);
        let margin_of_error = 3.58 * (truth * (1.0 - truth) / n_partitions as f64).sqrt();
        assert!(psm.view().data().iter().all(|prob| {
            *prob == 1.0 || (truth - margin_of_error < *prob && *prob < truth + margin_of_error)
        }));
    }

    #[test]
    fn test_pmf() {
        let n_items = 5;
        let mut permutation = Permutation::natural(n_items);
        let mut rng = thread_rng();
        permutation.shuffle(&mut rng);
        let mass = Mass::new(1.5);
        let discount = Discount::new(0.3);
        let mut ones = SquareMatrix::ones(n_items);
        let similarity = SimilarityBorrower(ones.view());
        let parameters = EPAParameters::new(&similarity, &permutation, mass, discount);
        let sum = Partition::iter(n_items)
            .map(|p| log_pmf(&mut Partition::from(&p[..]), &parameters).exp())
            .sum();
        assert!(0.9999999 <= sum, format!("{}", sum));
        assert!(sum <= 1.0000001, format!("{}", sum));
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__epa_partition(
    do_sampling: i32,
    n_partitions: i32,
    n_items: i32,
    partition_labels_ptr: *mut i32,
    partition_probs_ptr: *mut f64,
    seed_ptr: *const i32, // Assumed length is 32
    similarity_ptr: *mut f64,
    permutation_ptr: *const i32,
    mass: f64,
    discount: f64,
    use_random_permutation: i32,
) -> () {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let similarity = SimilarityBorrower(SquareMatrixBorrower::from_ptr(similarity_ptr, ni));
    let mut permutation = if use_random_permutation != 0 {
        Permutation::natural(ni)
    } else {
        let permutation_slice = slice::from_raw_parts(permutation_ptr, ni);
        let permutation_vector: Vec<usize> =
            permutation_slice.iter().map(|x| *x as usize).collect();
        Permutation::from_vector(permutation_vector).unwrap()
    };
    let mass = Mass::new(mass);
    let discount = Discount::new(discount);
    let matrix: &mut [i32] = slice::from_raw_parts_mut(partition_labels_ptr, np * ni);
    let probs: &mut [f64] = slice::from_raw_parts_mut(partition_probs_ptr, np);
    if do_sampling != 0 {
        let mut rng = &mut mk_rng_isaac(seed_ptr);
        for i in 0..np {
            if use_random_permutation != 0 {
                permutation.shuffle(&mut rng);
            }
            let parameters = EPAParameters::new(&similarity, &permutation, mass, discount);
            let p = engine(&parameters, TargetOrRandom::Random(rng));
            let labels = p.0.labels();
            for j in 0..ni {
                matrix[np * j + i] = i32::try_from(labels[j].unwrap()).unwrap();
            }
            probs[i] = p.1;
        }
    } else {
        for i in 0..np {
            let mut target_labels = Vec::with_capacity(ni);
            for j in 0..ni {
                target_labels.push(matrix[np * j + i]);
            }
            let mut target = Partition::from(&target_labels[..]);
            let parameters = EPAParameters::new(&similarity, &permutation, mass, discount);
            let p = engine::<IsaacRng>(&parameters, TargetOrRandom::Target(&mut target));
            probs[i] = p.1;
        }
    }
}
