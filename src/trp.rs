// Focal random partition distribution

use crate::clust::Clustering;
use crate::cpp::CPPParameters;
use crate::crp::CRPParameters;
use crate::epa::EPAParameters;
use crate::frp::FRPParameters;
use crate::lsp::LSPParameters;
use crate::mcmc::PriorLogWeight;
use crate::perm::Permutation;
use crate::prior::{PartitionLogProbability, PartitionSampler};
use crate::wgt::Weights;

use dahl_salso::clustering::{Clusterings, WorkingClustering};
use dahl_salso::log2cache::Log2Cache;
use dahl_salso::optimize::{
    BinderCMLossComputer, CMLossComputer, GeneralInformationBasedCMLossComputer,
    IDInformationBasedLoss, NIDInformationBasedLoss, NVIInformationBasedLoss, OMARICMLossComputer,
    VICMLossComputer,
};
use dahl_salso::LossFunction;
use rand::prelude::*;
use rand_isaac::IsaacRng;
use std::ffi::c_void;
use std::slice;

pub struct TRPParameters {
    pub target: Clustering,
    pub weights: Weights,
    pub permutation: Permutation,
    pub baseline_distribution: Box<dyn PartitionLogProbability>,
    pub loss_function: LossFunction,
}

impl TRPParameters {
    pub fn new(
        target: Clustering,
        weights: Weights,
        permutation: Permutation,
        baseline_distribution: Box<dyn PartitionLogProbability>,
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
                baseline_distribution,
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

fn compute_loss<'a>(x: &Clustering, parameters: &'a TRPParameters) -> f64 {
    let y: Vec<_> = x
        .allocation()
        .iter()
        .zip(parameters.target.allocation().iter())
        .filter(|&x| *x.0 != usize::max_value())
        .map(|x| (*x.0 as dahl_salso::LabelType, *x.1 as i32))
        .collect();
    let target_as_working = WorkingClustering::from_vector(
        y.iter().map(|x| x.0).collect(),
        (x.max_label() + 1) as dahl_salso::LabelType,
    );
    let cache = Log2Cache::new(match parameters.loss_function {
        LossFunction::VI(_) | LossFunction::NVI | LossFunction::ID | LossFunction::NID => {
            parameters.target.n_items()
        }
        _ => 0,
    });
    let loss_computer: Box<dyn CMLossComputer> = match parameters.loss_function {
        LossFunction::BinderDraws(a) => Box::new(BinderCMLossComputer::new(a)),
        LossFunction::OneMinusARI => Box::new(OMARICMLossComputer::new(1)),
        LossFunction::VI(a) => Box::new(VICMLossComputer::new(a, &cache)),
        LossFunction::NVI => Box::new(GeneralInformationBasedCMLossComputer::new(
            1,
            &cache,
            NVIInformationBasedLoss {},
        )),
        LossFunction::ID => Box::new(GeneralInformationBasedCMLossComputer::new(
            1,
            &cache,
            IDInformationBasedLoss {},
        )),
        LossFunction::NID => Box::new(GeneralInformationBasedCMLossComputer::new(
            1,
            &cache,
            NIDInformationBasedLoss {},
        )),
        _ => panic!("Unsupported loss function."),
    };
    let labels: Vec<_> = y.iter().map(|x| x.1).collect();
    let center_as_clusterings = Clusterings::from_i32_column_major_order(&labels[..], labels.len());
    loss_computer.compute_loss(
        &target_as_working,
        &center_as_clusterings.make_confusion_matrices(&target_as_working),
    )
}

fn engine<'a, T: Rng>(
    parameters: &'a TRPParameters,
    target: Option<&[usize]>,
    mut rng: Option<&mut T>,
) -> (Clustering, f64) {
    let ni = parameters.target.n_items();
    let mut log_probability = 0.0;
    let mut clustering = Clustering::unallocated(ni);
    for i in 0..clustering.n_items() {
        let ii = parameters.permutation.get(i);
        let scaled_weight = (i as f64) * parameters.weights[ii];
        let labels_and_weights = clustering
            .available_labels_for_allocation_with_target(target, ii)
            .map(|label| {
                let mut candidate = clustering.clone();
                candidate.allocate(ii, label);
                // This could be more efficient, since compute_loss does things every loop that could be done just once.
                let loss_value = compute_loss(&candidate, parameters);
                let weight = parameters.baseline_distribution.log_probability(&candidate)
                    - scaled_weight * loss_value;
                (label, weight)
            });
        let (subset_index, log_probability_contribution) = match &mut rng {
            Some(r) => clustering.select(labels_and_weights, true, 0, Some(r), true),
            None => clustering.select::<IsaacRng, _>(
                labels_and_weights,
                true,
                target.unwrap()[ii],
                None,
                true,
            ),
        };
        log_probability += log_probability_contribution;
        clustering.allocate(ii, subset_index);
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
    baseline_id: i32,
    baseline_ptr: *const c_void,
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
    let baseline_distribution: Box<dyn PartitionLogProbability> = match baseline_id {
        1 => {
            let p = std::ptr::NonNull::new(baseline_ptr as *mut CRPParameters).unwrap();
            Box::new(p.as_ref().clone())
        }
        2 => {
            let p = std::ptr::NonNull::new(baseline_ptr as *mut FRPParameters).unwrap();
            Box::new(p.as_ref().clone())
        }
        3 => {
            let p = std::ptr::NonNull::new(baseline_ptr as *mut LSPParameters).unwrap();
            Box::new(p.as_ref().clone())
        }
        4 => {
            let p = std::ptr::NonNull::new(baseline_ptr as *mut CPPParameters).unwrap();
            Box::new(p.as_ref().clone())
        }
        5 => {
            let p = std::ptr::NonNull::new(baseline_ptr as *mut EPAParameters).unwrap();
            Box::new(p.as_ref().clone())
        }
        _ => panic!("Unsupported prior ID: {}", baseline_id),
    };
    let loss_function = LossFunction::from_code(loss, a).unwrap();
    // First we create a new object.
    let obj = TRPParameters::new(
        focal,
        weights,
        permutation,
        baseline_distribution,
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
