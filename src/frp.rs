use dahl_partition::*;
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::*;
use std::convert::TryFrom;
use std::slice;

pub struct Weights(Vec<f64>);

impl Weights {
    pub fn zero(n_subsets: usize) -> Weights {
        Weights::constant(0.0, n_subsets)
    }

    pub fn constant(value: f64, n_subsets: usize) -> Weights {
        Weights(vec![value; n_subsets])
    }

    pub fn from(w: &[f64]) -> Option<Weights> {
        for ww in w.iter() {
            if ww.is_nan() || ww.is_infinite() || *ww < 0.0 {
                return None;
            }
        }
        Some(Weights(Vec::from(w)))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl std::ops::Index<usize> for Weights {
    type Output = f64;
    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}

enum TargetOrRandom<'a> {
    Target(&'a Partition),
    Random(ThreadRng),
}

pub fn engine(
    focal: &Partition,
    weights: &Weights,
    permutation: &Permutation,
    mass: f64,
    target: Option<&mut Partition>,
) -> (Partition, f64) {
    assert!(
        focal.is_canonical(),
        "Focal partition must be in canonical form."
    );
    assert_eq!(
        focal.n_subsets(),
        weights.len(),
        "Length of weights must equal the number of subsets of the focal partition."
    );
    assert_eq!(permutation.len(), focal.n_items());
    assert!(mass > 0.0, "Mass must be greater than 0.0.");
    let either = match target {
        Some(t) => {
            assert!(t.is_canonical());
            assert_eq!(t.n_items(), focal.n_items());
            t.canonicalize_by_permutation(Some(&permutation));
            TargetOrRandom::Target(t)
        }
        None => TargetOrRandom::Random(thread_rng()),
    };
    let ni = focal.n_items();
    let nsf = focal.n_subsets();

    let mut log_probability = 0.0;
    let mut partition = Partition::new(ni);
    let mut total_counter = vec![0.0; nsf];
    let mut intersection_counter = Vec::with_capacity(nsf);
    for _ in 0..nsf {
        intersection_counter.push(Vec::new())
    }
    for i in 0..ni {
        let ii = permutation[i];
        // Ensure there is an empty subset
        match partition.subsets().last() {
            None => partition.new_subset(),
            Some(last) => {
                if !last.is_empty() {
                    partition.new_subset()
                }
            }
        }
        let focal_subset_index = focal.label_of(ii).unwrap();
        let scaled_weight = if total_counter[focal_subset_index] == 0.0 {
            0.0
        } else {
            weights[focal_subset_index] / total_counter[focal_subset_index]
        };
        let probs: Vec<(usize, f64)> = partition
            .subsets()
            .iter()
            .enumerate()
            .map(|(subset_index, subset)| {
                let prob = if subset.is_empty() {
                    if total_counter[focal_subset_index] == 0.0 {
                        mass + weights[focal_subset_index]
                    } else {
                        mass
                    }
                } else {
                    (subset.n_items() as f64)
                        + scaled_weight * intersection_counter[focal_subset_index][subset_index]
                };
                (subset_index, prob)
            })
            .collect();
        let subset_index = match either {
            TargetOrRandom::Random(mut rng) => {
                let dist = WeightedIndex::new(probs.iter().map(|x| x.1)).unwrap();
                dist.sample(&mut rng)
            }
            TargetOrRandom::Target(t) => t.label_of(ii).unwrap(),
        };
        let numerator = probs[subset_index].1;
        let denominator = probs.iter().fold(0.0, |sum, x| sum + x.1);
        log_probability += (numerator / denominator).ln();
        if subset_index == intersection_counter[0].len() {
            for counter in intersection_counter.iter_mut() {
                counter.push(0.0);
            }
        }
        intersection_counter[focal_subset_index][subset_index] += 1.0;
        total_counter[focal_subset_index] += 1.0;
        partition.add_with_index(ii, subset_index);
    }
    partition.canonicalize();
    (partition, log_probability)
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
        let weights = Weights::zero(1);
        let mut permutation = Permutation::natural(n_items);
        let mut rng = thread_rng();
        for _ in 0..n_partitions {
            permutation.shuffle(&mut rng);
            samples.push_partition(&engine(&focal, &weights, &permutation, mass, None).0);
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
        let mass = 2.0;
        let mut permutation = Permutation::natural(n_items);
        let mut rng = thread_rng();
        for focal in Partition::iter(n_items) {
            permutation.shuffle(&mut rng);
            let focal = Partition::from(&focal[..]);
            let mut vec = Vec::with_capacity(focal.n_subsets());
            for _ in 0..focal.n_subsets() {
                vec.push(rng.gen_range(0.0, 10.0));
            }
            let weights = Weights::from(&vec[..]).unwrap();
            let sum = Partition::iter(n_items)
                .map(|p| {
                    engine(
                        &focal,
                        &weights,
                        &permutation,
                        mass,
                        Some(&mut Partition::from(&p[..])),
                    )
                    .1
                    .exp()
                })
                .sum();
            assert!(0.9999999 <= sum, format!("{}", sum));
            assert!(sum <= 1.0000001, format!("{}", sum));
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__focal_partition(
    n_partitions: i32,
    n_items: i32,
    focal_ptr: *const i32,
    weights_ptr: *const f64,
    permutation_ptr: *const i32,
    mass: f64,
    do_sampling: i32,
    use_random_permutations: i32,
    partition_labels_ptr: *mut i32,
    partition_probs_ptr: *mut f64,
) -> () {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let focal = Partition::from(slice::from_raw_parts(focal_ptr, ni));
    let weights = Weights::from(slice::from_raw_parts(weights_ptr, focal.n_subsets())).unwrap();
    let mut permutation = if use_random_permutations != 0 {
        Permutation::natural(ni)
    } else {
        let permutation_slice = slice::from_raw_parts(permutation_ptr, ni);
        let permutation_vector: Vec<usize> =
            permutation_slice.iter().map(|x| *x as usize).collect();
        Permutation::from_vector(permutation_vector).unwrap()
    };
    let matrix: &mut [i32] = slice::from_raw_parts_mut(partition_labels_ptr, np * ni);
    let probs: &mut [f64] = slice::from_raw_parts_mut(partition_probs_ptr, np);
    if do_sampling != 0 {
        let mut rng = thread_rng();
        for i in 0..np {
            if use_random_permutations != 0 {
                permutation.shuffle(&mut rng);
            }
            let p = engine(&focal, &weights, &permutation, mass, None);
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
            let p = engine(&focal, &weights, &permutation, mass, Some(&mut target));
            probs[i] = p.1;
        }
    }
}
