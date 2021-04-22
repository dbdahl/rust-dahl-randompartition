// Shrinkage partition distribution

use crate::clust::Clustering;
use crate::crp::CrpParameters;
use crate::distr::{
    FullConditional, PartitionSampler, PredictiveProbabilityFunction, ProbabilityMassFunction,
};
use crate::jlp::JlpParameters;
use crate::perm::Permutation;
use crate::shrink::Shrinkage;
use crate::up::UpParameters;

use dahl_salso::log2cache::Log2Cache;
use dahl_salso::LossFunction;
use rand::prelude::*;
use rand_isaac::IsaacRng;
use std::ffi::c_void;
use std::slice;

pub struct SpParameters {
    baseline_partition: Clustering,
    shrinkage: Shrinkage,
    permutation: Permutation,
    baseline_ppf: Box<dyn PredictiveProbabilityFunction>,
    loss_function: LossFunction,
    cache: Log2Cache,
}

impl SpParameters {
    pub fn new(
        baseline_partition: Clustering,
        shrinkage: Shrinkage,
        permutation: Permutation,
        baseline_ppf: Box<dyn PredictiveProbabilityFunction>,
        loss_function: LossFunction,
    ) -> Option<Self> {
        if (shrinkage.n_items() != baseline_partition.n_items())
            || (baseline_partition.n_items() != permutation.n_items())
        {
            None
        } else {
            let cache = Log2Cache::new(match loss_function {
                LossFunction::VI(_) | LossFunction::NVI | LossFunction::ID | LossFunction::NID => {
                    baseline_partition.n_items()
                }
                _ => 0,
            });
            Some(Self {
                baseline_partition: baseline_partition.standardize(),
                shrinkage,
                permutation,
                baseline_ppf,
                loss_function,
                cache,
            })
        }
    }

    pub fn shuffle_permutation<T: Rng>(&mut self, rng: &mut T) {
        self.permutation.shuffle(rng);
    }
}

fn expand_counts(counts: &mut Vec<Vec<usize>>, new_len: usize) {
    counts.iter_mut().map(|x| x.resize(new_len, 0)).collect()
}

impl FullConditional for SpParameters {
    // Implement starting only at item and subsequent items.
    fn log_full_conditional(&self, item: usize, clustering: &Clustering) -> Vec<(usize, f64)> {
        let mut target = clustering.allocation().clone();
        let candidate_labels = clustering.available_labels_for_reallocation(item);
        let mut partial_clustering = clustering.clone();
        for i in self.permutation.n_items_before(item)..partial_clustering.n_items() {
            partial_clustering.remove(self.permutation.get(i));
        }
        let mut marginal_counts = vec![0_usize; self.baseline_partition.max_label() + 1];
        let mut joint_counts = vec![vec![0_usize; 0]; self.baseline_partition.max_label() + 1];
        let max_label = partial_clustering.max_label();
        if max_label >= joint_counts[0].len() {
            expand_counts(&mut joint_counts, partial_clustering.max_label() + 1)
        }
        for i in 0..partial_clustering.n_items_allocated() {
            let item = self.permutation.get(i);
            let label_in_baseline = self.baseline_partition.get(item);
            let label = target[item];
            marginal_counts[label_in_baseline] += 1;
            joint_counts[label_in_baseline][label] += 1;
        }
        candidate_labels
            .map(|label| {
                target[item] = label;
                (
                    label,
                    engine::<IsaacRng>(
                        self,
                        partial_clustering.clone(),
                        marginal_counts.clone(),
                        joint_counts.clone(),
                        Some(&target[..]),
                        None,
                    )
                    .1,
                )
            })
            .collect()
    }
}

impl PartitionSampler for SpParameters {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering {
        engine_full(self, None, Some(rng)).0
    }
}

impl ProbabilityMassFunction for SpParameters {
    fn log_pmf(&self, partition: &Clustering) -> f64 {
        engine_full::<IsaacRng>(self, Some(partition.allocation()), None).1
    }
    fn is_normalized(&self) -> bool {
        true
    }
}

fn engine_full<'a, T: Rng>(
    parameters: &'a SpParameters,
    target: Option<&[usize]>,
    rng: Option<&mut T>,
) -> (Clustering, f64) {
    engine(
        parameters,
        Clustering::unallocated(parameters.baseline_partition.n_items()),
        vec![0_usize; parameters.baseline_partition.max_label() + 1],
        vec![vec![0_usize; 0]; parameters.baseline_partition.max_label() + 1],
        target,
        rng,
    )
}
fn engine<'a, T: Rng>(
    parameters: &'a SpParameters,
    mut clustering: Clustering,
    mut marginal_counts: Vec<usize>,
    mut joint_counts: Vec<Vec<usize>>,
    target: Option<&[usize]>,
    mut rng: Option<&mut T>,
) -> (Clustering, f64) {
    if let Ok(value) = std::env::var("DBD_ORIGINAL") {
        if value == "TRUE" {
            return engine_original(parameters, clustering, joint_counts, target, rng);
        }
    }
    let use_exponential_decay = match std::env::var("DBD_DECAY") {
        Ok(value) => value != "logistic",
        Err(_) => true,
    };
    let sill = if let Ok(value) = std::env::var("DBD_SILL") {
        value.parse().unwrap()
    } else {
        0.0
    };
    println!(
        "use_exponential_decay: {}, sill: {}",
        use_exponential_decay, sill
    );
    let (use_vi, a_plus_one) = match parameters.loss_function {
        LossFunction::BinderDraws(a) => (false, a + 1.0),
        LossFunction::VI(a) => (true, a + 1.0),
        _ => panic!("Unsupported loss function."),
    };
    let mut log_probability = 0.0;
    for i in clustering.n_items_allocated()..clustering.n_items() {
        let item = parameters.permutation.get(i);
        let candidate_labels: Vec<usize> = clustering
            .available_labels_for_allocation_with_target(target, item)
            .collect();
        let log_predictives =
            parameters
                .baseline_ppf
                .log_predictive(item, &candidate_labels, &clustering);
        let label_in_baseline = parameters.baseline_partition.get(item);
        let shrinkage = parameters.shrinkage[item];
        let max_candidate_label = *candidate_labels.iter().max().unwrap();
        if max_candidate_label >= joint_counts[label_in_baseline].len() {
            expand_counts(&mut joint_counts, max_candidate_label + 1)
        }
        let delta = |count: usize| {
            if !use_vi {
                count as f64
            } else {
                if count == 0 {
                    0.0
                } else {
                    let n1 = (count + 1) as f64;
                    let n0 = count as f64;
                    n1 * (n1.log2()) - n0 * (n0.log2())
                }
            }
        };
        let multiplier = if !use_vi {
            2.0 / (((i + 1) * (i + 1)) as f64)
        } else {
            1.0 / ((i + 1) as f64)
        };
        let distance_common = {
            let contribution = |c: usize, n: usize| {
                if !use_vi {
                    if c == 0 {
                        0.0
                    } else {
                        ((c as f64) / (n as f64)).powi(2)
                    }
                } else {
                    if c == 0 {
                        0.0
                    } else {
                        let x = (c as f64) / (n as f64);
                        x * x.ln()
                    }
                }
            };
            let x0 = multiplier * delta(marginal_counts[label_in_baseline]);
            let x1 = clustering.active_labels().iter().fold(0.0, |sum, label| {
                sum + contribution(clustering.size_of(*label), i + 1)
            });
            let x2 = marginal_counts
                .iter()
                .fold(0.0, |sum, c| sum + contribution(*c, i + 1));
            let x12 = joint_counts.iter().fold(0.0, |sum1, y| {
                sum1 + y.iter().fold(0.0, |sum2, c| sum2 + contribution(*c, i + 1))
            });
            (a_plus_one - 1.0) * (x0 + x1) + x2 - a_plus_one * x12
        };
        let distances = candidate_labels.iter().map(|label| {
            let nm = clustering.size_of(*label);
            let nj = joint_counts[label_in_baseline][*label];
            let distance_delta = multiplier * (delta(nm) - a_plus_one * delta(nj));
            distance_common + distance_delta
        });
        let distances: Vec<_> = distances.collect();
        let sum_of_distances: f64 = distances.iter().sum();
        let sum_of_distances = if sum_of_distances > 0.0 {
            sum_of_distances
        } else {
            1.0
        };
        let normalized_distances = distances.iter().map(|x| x / sum_of_distances);
        let labels_and_log_weights = log_predictives.iter().zip(normalized_distances).map(
            |((label, log_probability), normalized_distance)| {
                let log_weight = if use_exponential_decay {
                    *log_probability - shrinkage * normalized_distance
                } else {
                    *log_probability - (1.0 + (shrinkage * normalized_distance - sill).exp()).ln()
                };
                (*label, log_weight)
            },
        );
        let (label, log_probability_contribution) = match &mut rng {
            Some(r) => clustering.select(labels_and_log_weights, true, 0, Some(r), true),
            None => clustering.select::<IsaacRng, _>(
                labels_and_log_weights,
                true,
                target.unwrap()[item],
                None,
                true,
            ),
        };
        clustering.allocate(item, label);
        log_probability += log_probability_contribution;
        marginal_counts[label_in_baseline] += 1;
        joint_counts[label_in_baseline][label] += 1;
    }
    (clustering, log_probability)
}

fn engine_original<'a, T: Rng>(
    parameters: &'a SpParameters,
    mut clustering: Clustering,
    mut counts: Vec<Vec<usize>>,
    target: Option<&[usize]>,
    mut rng: Option<&mut T>,
) -> (Clustering, f64) {
    let (use_vi, a_plus_one) = match parameters.loss_function {
        LossFunction::BinderDraws(a) => (false, a + 1.0),
        LossFunction::VI(a) => (true, a + 1.0),
        _ => panic!("Unsupported loss function."),
    };
    let mut log_probability = 0.0;
    for i in clustering.n_items_allocated()..clustering.n_items() {
        let item = parameters.permutation.get(i);
        let label_in_baseline = parameters.baseline_partition.get(item);
        let scaled_shrinkage = ((i + 1) as f64) * parameters.shrinkage[item];
        let candidate_labels: Vec<usize> = clustering
            .available_labels_for_allocation_with_target(target, item)
            .collect();
        let max_candidate_label = *candidate_labels.iter().max().unwrap();
        if max_candidate_label >= counts[label_in_baseline].len() {
            expand_counts(&mut counts, max_candidate_label + 1)
        }
        let multiplier = if !use_vi {
            2.0 / (((i + 1) * (i + 1)) as f64)
        } else {
            1.0 / ((i + 1) as f64)
        };
        let labels_and_log_weights = parameters
            .baseline_ppf
            .log_predictive(item, &candidate_labels, &clustering)
            .into_iter()
            .map(|(label, log_probability)| {
                let nm = clustering.size_of(label);
                let nj = counts[label_in_baseline][label];
                let distance = if !use_vi {
                    // Binder loss
                    fn binder_delta(count: usize) -> f64 {
                        count as f64
                    }
                    multiplier * (binder_delta(nm) - a_plus_one * binder_delta(nj))
                } else {
                    // Variation of information loss
                    // Since this is a function on integers, we could cache these calculations for more computational efficiency.
                    fn vi_delta(count: usize) -> f64 {
                        if count == 0 {
                            0.0
                        } else {
                            let n1 = (count + 1) as f64;
                            let n0 = count as f64;
                            n1 * (n1.log2()) - n0 * (n0.log2())
                        }
                    }
                    multiplier * (vi_delta(nm) - a_plus_one * vi_delta(nj))
                };
                (label, log_probability - scaled_shrinkage * distance)
            });
        let (label, log_probability_contribution) = match &mut rng {
            Some(r) => clustering.select(labels_and_log_weights, true, 0, Some(r), true),
            None => clustering.select::<IsaacRng, _>(
                labels_and_log_weights,
                true,
                target.unwrap()[item],
                None,
                true,
            ),
        };
        log_probability += log_probability_contribution;
        clustering.allocate(item, label);
        counts[label_in_baseline][label] += 1;
    }
    (clustering, log_probability)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_goodness_of_fit_constructive() {
        let n_items = 4;
        let discount = 0.1;
        let mass = Mass::new_with_variable_constraint(2.0, discount);
        let discount = Discount::new(discount);
        let loss_function = LossFunction::VI(1.0);
        let mut rng = thread_rng();
        for target in Clustering::iter(n_items) {
            let target = Clustering::from_vector(target);
            let mut vec = Vec::with_capacity(target.n_clusters());
            for _ in 0..target.n_items() {
                vec.push(rng.gen_range(0.0..10.0));
            }
            let shrinkage = Shrinkage::from(&vec[..]).unwrap();
            let permutation = Permutation::random(n_items, &mut rng);
            let baseline_distribution =
                CrpParameters::new_with_mass_and_discount(n_items, mass, discount);
            let parameters = SpParameters::new(
                target,
                shrinkage,
                permutation,
                Box::new(baseline_distribution),
                loss_function,
            )
            .unwrap();
            let sample_closure = || parameters.sample(&mut thread_rng());
            let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
            crate::testing::assert_goodness_of_fit(
                10000,
                n_items,
                sample_closure,
                log_prob_closure,
                1,
                0.001,
            );
        }
    }

    #[test]
    fn test_pmf() {
        let n_items = 5;
        let discount = 0.1;
        let mass = Mass::new_with_variable_constraint(2.0, discount);
        let discount = Discount::new(discount);
        let loss_function = LossFunction::VI(1.0);
        let mut rng = thread_rng();
        for target in Clustering::iter(n_items) {
            let target = Clustering::from_vector(target);
            let mut vec = Vec::with_capacity(target.n_clusters());
            for _ in 0..target.n_items() {
                vec.push(rng.gen_range(0.0..10.0));
            }
            let shrinkage = Shrinkage::from(&vec[..]).unwrap();
            let permutation = Permutation::random(n_items, &mut rng);
            let baseline_distribution =
                CrpParameters::new_with_mass_and_discount(n_items, mass, discount);
            let parameters = SpParameters::new(
                target,
                shrinkage,
                permutation,
                Box::new(baseline_distribution),
                loss_function,
            )
            .unwrap();
            let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
            crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__spparameters_new(
    n_items: i32,
    baseline_partition_ptr: *const i32,
    shrinkage_ptr: *const f64,
    permutation_ptr: *const i32,
    use_natural_permutation: i32,
    baseline_distr_id: i32,
    baseline_distr_ptr: *const c_void,
    loss: i32,
    a: f64,
) -> *mut SpParameters {
    let ni = n_items as usize;
    let opined = Clustering::from_slice(slice::from_raw_parts(baseline_partition_ptr, ni));
    let shrinkage = Shrinkage::from(slice::from_raw_parts(shrinkage_ptr, ni)).unwrap();
    let permutation = if use_natural_permutation != 0 {
        Permutation::natural_and_fixed(ni)
    } else {
        let permutation_slice = slice::from_raw_parts(permutation_ptr, ni);
        let permutation_vector: Vec<usize> =
            permutation_slice.iter().map(|x| *x as usize).collect();
        Permutation::from_vector(permutation_vector).unwrap()
    };
    let loss_function = LossFunction::from_code(loss, a).unwrap();
    let obj = match baseline_distr_id {
        1 => {
            let p = std::ptr::NonNull::new(baseline_distr_ptr as *mut CrpParameters).unwrap();
            let baseline_distribution = p.as_ref().clone();
            SpParameters::new(
                opined,
                shrinkage,
                permutation,
                Box::new(baseline_distribution),
                loss_function,
            )
            .unwrap()
        }
        7 => {
            let p = std::ptr::NonNull::new(baseline_distr_ptr as *mut UpParameters).unwrap();
            let baseline_distribution = p.as_ref().clone();
            SpParameters::new(
                opined,
                shrinkage,
                permutation,
                Box::new(baseline_distribution),
                loss_function,
            )
            .unwrap()
        }
        8 => {
            let p = std::ptr::NonNull::new(baseline_distr_ptr as *mut JlpParameters).unwrap();
            let baseline_distribution = p.as_ref().clone();
            SpParameters::new(
                opined,
                shrinkage,
                permutation,
                Box::new(baseline_distribution),
                loss_function,
            )
            .unwrap()
        }
        _ => panic!("Unsupported prior ID: {}", baseline_distr_id),
    };
    // First we create a new object.
    // Then copy it to the heap (so we have a stable pointer to it).
    let boxed_obj = Box::new(obj);
    // Then return a pointer by converting our `Box<_>` into a raw pointer
    Box::into_raw(boxed_obj)
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__spparameters_free(obj: *mut SpParameters) {
    // As a rule of thumb, freeing a null pointer is just a noop.
    if obj.is_null() {
        return;
    }
    // Convert the raw pointer back to a Box<_>
    let boxed = Box::from_raw(obj);
    // Then explicitly drop it (optional)
    drop(boxed);
}
