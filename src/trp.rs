// Focal random partition distribution

use crate::clust::Clustering;
use crate::crp::CRPParameters;
use crate::mcmc::PriorLogWeight;
use crate::perm::Permutation;
use crate::prior::{PartitionLogProbability, PartitionSampler};
use crate::wgt::Weights;

use crate::cpp::CPPParameters;
use crate::epa::EPAParameters;
use crate::frp::FRPParameters;
use crate::lsp::LSPParameters;
use dahl_salso::LossFunction;
use rand::prelude::*;
use rand_isaac::IsaacRng;
use std::ffi::c_void;
use std::slice;

pub struct TRPParameters {
    pub target: Clustering,
    pub weights: Weights,
    pub permutation: Permutation,
    pub base_distribution: Box<dyn PartitionLogProbability>,
    pub loss_function: LossFunction,
}

impl TRPParameters {
    pub fn new(
        target: Clustering,
        weights: Weights,
        permutation: Permutation,
        base_distribution: Box<dyn PartitionLogProbability>,
        loss_function: LossFunction,
    ) -> Option<Self> {
        if weights.len() != target.n_items() {
            None
        } else if target.n_items() != permutation.len() {
            None
        } else {
            Some(Self {
                target: target.standardize(),
                weights,
                permutation,
                base_distribution,
                loss_function,
            })
        }
    }

    pub fn shuffle_permutation<T: Rng>(&mut self, rng: &mut T) {
        self.permutation.shuffle(rng);
    }
}

impl PriorLogWeight for TRPParameters {
    fn log_weight(&self, item_index: usize, subset_index: usize, clustering: &Clustering) -> f64 {
        let mut p = clustering.allocation().clone();
        p[item_index] = subset_index;
        engine::<IsaacRng>(self, Some(&p[..]), None).1
    }
}

impl PartitionSampler for TRPParameters {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering {
        engine(self, None, Some(rng)).0
    }
}

impl PartitionLogProbability for TRPParameters {
    fn log_probability(&self, partition: &Clustering) -> f64 {
        engine::<IsaacRng>(self, Some(partition.allocation()), None).1
    }
    fn is_normalized(&self) -> bool {
        true
    }
}

fn engine<'a, T: Rng>(
    parameters: &'a TRPParameters,
    _target: Option<&[usize]>,
    mut _rng: Option<&mut T>,
) -> (Clustering, f64) {
    /*
    let ni = parameters.target.n_items();
    let mass = parameters.mass.unwrap();
    let discount = parameters.discount.unwrap();
    let power = parameters.power.unwrap();
    let mut log_probability = 0.0;
    let mut clustering = Clustering::unallocated(ni);
    let mut total_counter = vec![0.0; parameters.focal.max_label() + 1];
    let mut intersection_counter = Vec::with_capacity(total_counter.len());
    for _ in 0..total_counter.len() {
        intersection_counter.push(Vec::new())
    }
    for i in 0..clustering.n_items() {
        let ii = parameters.permutation.get(i);
        let focal_subset_index = parameters.focal[ii];
        let scaled_weight = (i as f64) * parameters.weights[ii];
        let normalized_scaled_weight = if total_counter[focal_subset_index] == 0.0 {
            0.0
        } else {
            scaled_weight / total_counter[focal_subset_index]
        };
        let n_occupied_subsets = clustering.n_clusters() as f64;
        let labels_and_weights = clustering
            .available_labels_for_allocation_with_target(target, ii)
            .map(|label| {
                let n_items_in_cluster = clustering.size_of(label);
                let weight = if n_items_in_cluster == 0 {
                    if n_occupied_subsets == 0.0 {
                        1.0
                    } else {
                        mass + discount * n_occupied_subsets + {
                            if total_counter[focal_subset_index] == 0.0 {
                                scaled_weight
                            } else {
                                0.0
                            }
                        }
                    }
                } else {
                    ((n_items_in_cluster as f64) - discount).powf(power)
                        + normalized_scaled_weight * intersection_counter[focal_subset_index][label]
                };
                (label, weight)
            });
        let (subset_index, log_probability_contribution) = match &mut rng {
            Some(r) => clustering.select(labels_and_weights, false, 0, Some(r), true),
            None => clustering.select::<IsaacRng, _>(
                labels_and_weights,
                false,
                target.unwrap()[ii],
                None,
                true,
            ),
        };
        log_probability += log_probability_contribution;
        if subset_index >= intersection_counter[0].len() {
            for counter in intersection_counter.iter_mut() {
                counter.resize(subset_index + 1, 0.0);
            }
        }
        intersection_counter[focal_subset_index][subset_index] += 1.0;
        total_counter[focal_subset_index] += 1.0;
        clustering.allocate(ii, subset_index);
    }
    */
    let clustering = parameters.target.clone();
    let log_probability = 0.0;
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
                vec.push(rng.gen_range(0.0, 10.0));
            }
            let weights = Weights::from(&vec[..]).unwrap();
            let permutation = Permutation::random(n_items, &mut rng);
            let parameters = TRPParameters::new(
                target,
                weights,
                permutation,
                Box::new(CRPParameters::new_with_mass_and_discount(
                    mass, discount, n_items,
                )),
                loss_function,
            )
            .unwrap();
            let sample_closure = || parameters.sample(&mut thread_rng());
            let log_prob_closure =
                |clustering: &mut Clustering| parameters.log_probability(clustering);
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
                vec.push(rng.gen_range(0.0, 10.0));
            }
            let weights = Weights::from(&vec[..]).unwrap();
            let permutation = Permutation::random(n_items, &mut rng);
            let parameters = TRPParameters::new(
                target,
                weights,
                permutation,
                Box::new(CRPParameters::new_with_mass_and_discount(
                    mass, discount, n_items,
                )),
                loss_function,
            )
            .unwrap();
            let log_prob_closure =
                |clustering: &mut Clustering| parameters.log_probability(clustering);
            crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__trpparameters_new(
    n_items: i32,
    focal_ptr: *const i32,
    weights_ptr: *const f64,
    permutation_ptr: *const i32,
    use_natural_permutation: i32,
    base_id: i32,
    base_ptr: *const c_void,
    loss: i32,
    a: f64,
) -> *mut TRPParameters {
    let ni = n_items as usize;
    let focal = Clustering::from_slice(slice::from_raw_parts(focal_ptr, ni));
    let weights = Weights::from(slice::from_raw_parts(weights_ptr, ni)).unwrap();
    let permutation = if use_natural_permutation != 0 {
        Permutation::natural_and_fixed(ni)
    } else {
        let permutation_slice = slice::from_raw_parts(permutation_ptr, ni);
        let permutation_vector: Vec<usize> =
            permutation_slice.iter().map(|x| *x as usize).collect();
        Permutation::from_vector(permutation_vector).unwrap()
    };
    let base_distribution: Box<dyn PartitionLogProbability> = match base_id {
        1 => {
            let p = std::ptr::NonNull::new(base_ptr as *mut CRPParameters).unwrap();
            Box::new(p.as_ref().clone())
        }
        2 => {
            let p = std::ptr::NonNull::new(base_ptr as *mut FRPParameters).unwrap();
            Box::new(p.as_ref().clone())
        }
        3 => {
            let p = std::ptr::NonNull::new(base_ptr as *mut LSPParameters).unwrap();
            Box::new(p.as_ref().clone())
        }
        4 => {
            let p = std::ptr::NonNull::new(base_ptr as *mut CPPParameters).unwrap();
            Box::new(p.as_ref().clone())
        }
        5 => {
            let p = std::ptr::NonNull::new(base_ptr as *mut EPAParameters).unwrap();
            Box::new(p.as_ref().clone())
        }
        _ => panic!("Unsupported prior ID: {}", base_id),
    };
    let loss_function = LossFunction::from_code(loss, a).unwrap();
    // First we create a new object.
    let obj = TRPParameters::new(
        focal,
        weights,
        permutation,
        base_distribution,
        loss_function,
    )
    .unwrap();
    // Then copy it to the heap (so we have a stable pointer to it).
    let boxed_obj = Box::new(obj);
    // Then return a pointer by converting our `Box<_>` into a raw pointer
    Box::into_raw(boxed_obj)
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__trpparameters_free(obj: *mut TRPParameters) {
    // As a rule of thumb, freeing a null pointer is just a noop.
    if obj.is_null() {
        return;
    }
    // Convert the raw pointer back to a Box<_>
    let boxed = Box::from_raw(obj);
    // Then explicitly drop it (optional)
    drop(boxed);
}
