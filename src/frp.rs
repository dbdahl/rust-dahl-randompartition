extern crate dahl_partition;
extern crate dahl_salso;
extern crate rand;

use dahl_partition::*;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use rand::Rng;
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

fn mk_intersection_counter(n_subsets: usize) -> Vec<Vec<f64>> {
    let mut counter = Vec::with_capacity(n_subsets);
    for _ in 0..n_subsets {
        counter.push(Vec::new())
    }
    counter
}

enum TargetOrRandom<'a> {
    Target(&'a Partition),
    Random(rand::prelude::ThreadRng),
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
    let mut intersection_counter = mk_intersection_counter(nsf);
    for i in 0..ni {
        let ii = permutation[i];
        ensure_empty_subset(&mut partition);
        let focal_subset_index = focal.label_of(ii).unwrap();
        let constant = if total_counter[focal_subset_index] == 0.0 {
            0.0
        } else {
            weights[focal_subset_index] / total_counter[focal_subset_index]
        };
        let probs = partition
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
                        + constant * intersection_counter[focal_subset_index][subset_index]
                };
                (subset_index, prob)
            });
        let subset_index = match either {
            TargetOrRandom::Random(mut rng) => {
                let dist = WeightedIndex::new(probs.map(|x| x.1)).unwrap();
                dist.sample(&mut rng)
            }
            TargetOrRandom::Target(t) => {
                let index = t.label_of(ii).unwrap();
                let mut numerator = -1.0;
                let denominator = probs.fold(0.0, |sum, x| {
                    if x.0 == index {
                        numerator = x.1;
                    }
                    sum + x.1
                });
                log_probability += (numerator/denominator).ln();
                index
            }
        };
        if subset_index == intersection_counter[0].len() {
            for counter in intersection_counter.iter_mut() {
                counter.push(0.0);
            }
        }
        intersection_counter[focal_subset_index][subset_index] += 1.0;
        total_counter[focal_subset_index] += 1.0;
        for fi in 0..focal.n_subsets() {
            assert_eq!(
                intersection_counter[fi].iter().fold(0.0, |sum, x| sum + *x),
                total_counter[fi]
            );
        }
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
        let n_items = 4;
        let mass = 3.0;
        let mut permutation = Permutation::natural(n_items);
        let mut rng = thread_rng();
        for focal in Partition::iter(n_items) {
            permutation.shuffle(&mut rng);
            let focal = Partition::from(&focal[..]);
            let weights = Weights::constant(2.0, focal.n_subsets());
            //let mut vec = Vec::with_capacity(focal.n_subsets());
            //for _ in 0..focal.n_subsets() {
            //    vec.push(rng.gen_range(0.0, 10.0));
            //}
            //let weights = Weights::from(&vec[..]).unwrap();
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
pub unsafe extern "C" fn dahl_randompartition__rfp__sample(
    n_partitions: i32,
    n_items: i32,
    focal_ptr: *const i32,
    weights_ptr: *const f64,
    permutation_ptr: *const i32,
    mass: f64,
    ptr: *mut i32,
) -> () {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let array: &mut [i32] = slice::from_raw_parts_mut(ptr, np * ni);
    let focal = Partition::from(slice::from_raw_parts(focal_ptr, ni));
    let weights = Weights::from(slice::from_raw_parts(weights_ptr, focal.n_subsets())).unwrap();
    let permutation_slice = slice::from_raw_parts(permutation_ptr, ni);
    let random_permutation = permutation_slice[0] == -1;
    let mut permutation = if !random_permutation {
        let permutation_vector: Vec<usize> =
            permutation_slice.iter().map(|x| *x as usize).collect();
        Permutation::from_vector(permutation_vector).unwrap()
    } else {
        Permutation::natural(ni)
    };
    let mut rng = thread_rng();
    for i in 0..np {
        if random_permutation {
            permutation.shuffle(&mut rng);
        }
        let p = engine(&focal, &weights, &permutation, mass, None);
        let labels = p.0.labels();
        for j in 0..ni {
            array[np * j + i] = i32::try_from(labels[j].unwrap()).unwrap();
        }
    }
}
