// Shrinkage partition distribution

use crate::clust::Clustering;
use crate::cpp::CppParameters;
use crate::crp::CrpParameters;
use crate::distr::{PartitionSampler, PredictiveProbabilityFunctionOld};
use crate::epa::EpaParameters;
use crate::frp::FrpParameters;
use crate::lsp::LspParameters;
use crate::perm::Permutation;
use crate::prior::PartitionLogProbability;
use crate::up::UpParameters;
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

pub struct SpParameters {
    pub baseline_partition: Clustering,
    pub weights: Weights,
    pub permutation: Permutation,
    pub baseline_distribution: Box<dyn PredictiveProbabilityFunctionOld>,
    pub loss_function: LossFunction,
    cache: Log2Cache,
}

impl SpParameters {
    pub fn new(
        target: Clustering,
        weights: Weights,
        permutation: Permutation,
        baseline_distribution: Box<dyn PredictiveProbabilityFunctionOld>,
        loss_function: LossFunction,
    ) -> Option<Self> {
        if (weights.n_items() != target.n_items()) || (target.n_items() != permutation.n_items()) {
            None
        } else {
            let cache = Log2Cache::new(match loss_function {
                LossFunction::VI(_) | LossFunction::NVI | LossFunction::ID | LossFunction::NID => {
                    target.n_items()
                }
                _ => 0,
            });
            Some(Self {
                baseline_partition: target.standardize(),
                weights,
                permutation,
                baseline_distribution,
                loss_function,
                cache,
            })
        }
    }

    pub fn shuffle_permutation<T: Rng>(&mut self, rng: &mut T) {
        self.permutation.shuffle(rng);
    }
}

impl PredictiveProbabilityFunctionOld for SpParameters {
    fn log_predictive_probability(
        &self,
        item_index: usize,
        subset_index: usize,
        clustering: &Clustering,
    ) -> f64 {
        let mut p = clustering.allocation().clone();
        p[item_index] = subset_index;
        engine::<IsaacRng>(self, Some(&p[..]), None).1
    }
}

impl PartitionSampler for SpParameters {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering {
        engine(self, None, Some(rng)).0
    }
}

impl PartitionLogProbability for SpParameters {
    fn log_probability(&self, partition: &Clustering) -> f64 {
        engine::<IsaacRng>(self, Some(partition.allocation()), None).1
    }
    fn is_normalized(&self) -> bool {
        true
    }
}

fn compute_loss<'a, 'b>(
    x: &Clustering,
    parameters: &'a SpParameters,
    loss_computer: &Box<dyn CMLossComputer + 'b>,
) -> f64 {
    let y: Vec<_> = x
        .allocation()
        .iter()
        .zip(parameters.baseline_partition.allocation().iter())
        .filter(|&x| *x.0 != usize::MAX)
        .map(|x| (*x.0 as dahl_salso::LabelType, *x.1 as i32))
        .collect();
    let target_as_working = WorkingClustering::from_vector(
        y.iter().map(|x| x.0).collect(),
        (x.max_label() + 1) as dahl_salso::LabelType,
    );
    let labels: Vec<_> = y.iter().map(|x| x.1).collect();
    let opined_as_clusterings = Clusterings::from_i32_column_major_order(&labels[..], labels.len());
    loss_computer.compute_loss(
        &target_as_working,
        &opined_as_clusterings.make_confusion_matrices(&target_as_working),
    )
}

fn engine<'a, T: Rng>(
    parameters: &'a SpParameters,
    target: Option<&[usize]>,
    mut rng: Option<&mut T>,
) -> (Clustering, f64) {
    let ni = parameters.baseline_partition.n_items();
    let loss_computer: Box<dyn CMLossComputer> = match parameters.loss_function {
        LossFunction::BinderDraws(a) => Box::new(BinderCMLossComputer::new(a)),
        LossFunction::OneMinusARI => Box::new(OMARICMLossComputer::new(1)),
        LossFunction::VI(a) => Box::new(VICMLossComputer::new(a, &parameters.cache)),
        LossFunction::NVI => Box::new(GeneralInformationBasedCMLossComputer::new(
            1,
            &parameters.cache,
            NVIInformationBasedLoss {},
        )),
        LossFunction::ID => Box::new(GeneralInformationBasedCMLossComputer::new(
            1,
            &parameters.cache,
            IDInformationBasedLoss {},
        )),
        LossFunction::NID => Box::new(GeneralInformationBasedCMLossComputer::new(
            1,
            &parameters.cache,
            NIDInformationBasedLoss {},
        )),
        _ => panic!("Unsupported loss function."),
    };
    let mut log_probability = 0.0;
    let mut clustering = Clustering::unallocated(ni);
    for i in 0..clustering.n_items() {
        let ii = parameters.permutation.get(i);
        let scaled_weight = ((i + 1) as f64) * parameters.weights[ii];
        let available_labels = clustering
            .available_labels_for_allocation_with_target(target, ii)
            .collect::<Vec<_>>();
        clustering.allocate(ii, clustering.new_label());
        let labels_and_weights = available_labels
            .into_iter()
            .map(|label| {
                clustering.reallocate(ii, label);
                let loss_value = compute_loss(&clustering, parameters, &loss_computer);
                let weight = parameters.baseline_distribution.log_predictive_probability(
                    ii,
                    label,
                    &clustering,
                ) - scaled_weight * loss_value;
                (label, weight)
            })
            .collect::<Vec<_>>()
            .into_iter();
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
        clustering.reallocate(ii, subset_index);
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
            let weights = Weights::from(&vec[..]).unwrap();
            let permutation = Permutation::random(n_items, &mut rng);
            let parameters = SpParameters::new(
                target,
                weights,
                permutation,
                Box::new(CrpParameters::new_with_mass_and_discount(
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
                vec.push(rng.gen_range(0.0..10.0));
            }
            let weights = Weights::from(&vec[..]).unwrap();
            let permutation = Permutation::random(n_items, &mut rng);
            let parameters = SpParameters::new(
                target,
                weights,
                permutation,
                Box::new(CrpParameters::new_with_mass_and_discount(
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
    baseline_partition_ptr: *const i32,
    weights_ptr: *const f64,
    permutation_ptr: *const i32,
    use_natural_permutation: i32,
    baseline_distr_id: i32,
    baseline_distr_ptr: *const c_void,
    loss: i32,
    a: f64,
) -> *mut SpParameters {
    let ni = n_items as usize;
    let opined = Clustering::from_slice(slice::from_raw_parts(baseline_partition_ptr, ni));
    let weights = Weights::from(slice::from_raw_parts(weights_ptr, ni)).unwrap();
    let permutation = if use_natural_permutation != 0 {
        Permutation::natural_and_fixed(ni)
    } else {
        let permutation_slice = slice::from_raw_parts(permutation_ptr, ni);
        let permutation_vector: Vec<usize> =
            permutation_slice.iter().map(|x| *x as usize).collect();
        Permutation::from_vector(permutation_vector).unwrap()
    };
    let baseline_distribution: Box<dyn PredictiveProbabilityFunctionOld> = match baseline_distr_id {
        1 => {
            let p = std::ptr::NonNull::new(baseline_distr_ptr as *mut CrpParameters).unwrap();
            Box::new(p.as_ref().clone())
        }
        2 => {
            let p = std::ptr::NonNull::new(baseline_distr_ptr as *mut FrpParameters).unwrap();
            Box::new(p.as_ref().clone())
        }
        3 => {
            let p = std::ptr::NonNull::new(baseline_distr_ptr as *mut LspParameters).unwrap();
            Box::new(p.as_ref().clone())
        }
        4 => {
            let p = std::ptr::NonNull::new(baseline_distr_ptr as *mut CppParameters).unwrap();
            Box::new(p.as_ref().clone())
        }
        5 => {
            let p = std::ptr::NonNull::new(baseline_distr_ptr as *mut EpaParameters).unwrap();
            Box::new(p.as_ref().clone())
        }
        //        6 => {
        //            let p = std::ptr::NonNull::new(baseline_distr_ptr as *mut TRPParameters).unwrap();
        //            Box::new(p.as_ref().clone())
        //        }
        7 => {
            let p = std::ptr::NonNull::new(baseline_distr_ptr as *mut UpParameters).unwrap();
            Box::new(p.as_ref().clone())
        }
        _ => panic!("Unsupported prior ID: {}", baseline_distr_id),
    };
    let loss_function = LossFunction::from_code(loss, a).unwrap();
    // First we create a new object.
    let obj = SpParameters::new(
        opined,
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
pub unsafe extern "C" fn dahl_randompartition__trpparameters_free(obj: *mut SpParameters) {
    // As a rule of thumb, freeing a null pointer is just a noop.
    if obj.is_null() {
        return;
    }
    // Convert the raw pointer back to a Box<_>
    let boxed = Box::from_raw(obj);
    // Then explicitly drop it (optional)
    drop(boxed);
}
