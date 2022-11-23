// Old shrinkage partition distribution (based on a partition distance function, i.e., VI or Binder)

use crate::clust::Clustering;
use crate::distr::{
    FullConditional, HasPermutation, HasVectorShrinkage, NormalizedProbabilityMassFunction,
    PartitionSampler, PredictiveProbabilityFunction, ProbabilityMassFunction,
};
use crate::perm::Permutation;
use crate::shrink::Shrinkage;

use dahl_salso::log2cache::Log2Cache;
use dahl_salso::LossFunction;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;

#[derive(Debug, Clone)]
pub struct OldSpParameters<D: PredictiveProbabilityFunction + Clone> {
    pub baseline_partition: Clustering,
    pub shrinkage: Shrinkage,
    pub permutation: Permutation,
    baseline_ppf: D,
    loss_function: LossFunction,
    scaling_exponent: f64,
    cache: Log2Cache,
}

impl<D: PredictiveProbabilityFunction + Clone> OldSpParameters<D> {
    pub fn new(
        baseline_partition: Clustering,
        shrinkage: Shrinkage,
        permutation: Permutation,
        baseline_ppf: D,
        use_vi: bool,
        a: f64,
        scaling_exponent: f64,
    ) -> Option<Self> {
        if (shrinkage.n_items() != baseline_partition.n_items())
            || (baseline_partition.n_items() != permutation.n_items())
        {
            None
        } else {
            let cache = Log2Cache::new(if use_vi {
                baseline_partition.n_items()
            } else {
                0
            });
            if a <= 0.0 {
                return None;
            }
            let loss_function = match use_vi {
                true => LossFunction::VI(a),
                false => LossFunction::BinderDraws(a),
            };
            Some(Self {
                baseline_partition: baseline_partition.standardize(),
                shrinkage,
                permutation,
                baseline_ppf,
                loss_function,
                scaling_exponent,
                cache,
            })
        }
    }
}

fn expand_counts(counts: &mut Vec<Vec<usize>>, new_len: usize) {
    counts.iter_mut().map(|x| x.resize(new_len, 0)).collect()
}

impl<D: PredictiveProbabilityFunction + Clone> FullConditional for OldSpParameters<D> {
    // Implement starting only at item and subsequent items.
    fn log_full_conditional(&self, item: usize, clustering: &Clustering) -> Vec<(usize, f64)> {
        let mut target = clustering.allocation().clone();
        let candidate_labels = clustering.available_labels_for_reallocation(item);
        let mut partial_clustering = clustering.clone();
        for i in self.permutation.n_items_before(item)..partial_clustering.n_items() {
            partial_clustering.remove(self.permutation.get(i));
        }
        let mut counts = vec![vec![0_usize; 0]; self.baseline_partition.max_label() + 1];
        let max_label = partial_clustering.max_label();
        if max_label >= counts[0].len() {
            expand_counts(&mut counts, partial_clustering.max_label() + 1)
        }
        for i in 0..partial_clustering.n_items_allocated() {
            let item = self.permutation.get(i);
            let label_in_baseline = self.baseline_partition.get(item);
            let label = target[item];
            counts[label_in_baseline][label] += 1;
        }
        candidate_labels
            .map(|label| {
                target[item] = label;
                (
                    label,
                    engine::<D, Pcg64Mcg>(
                        self,
                        partial_clustering.clone(),
                        counts.clone(),
                        Some(&target[..]),
                        None,
                    )
                    .1,
                )
            })
            .collect()
    }
}

impl<D: PredictiveProbabilityFunction + Clone> PartitionSampler for OldSpParameters<D> {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering {
        engine_full(self, None, Some(rng)).0
    }
}

impl<D: PredictiveProbabilityFunction + Clone> ProbabilityMassFunction for OldSpParameters<D> {
    fn log_pmf(&self, partition: &Clustering) -> f64 {
        engine_full::<D, Pcg64Mcg>(self, Some(partition.allocation()), None).1
    }
}

impl<D: PredictiveProbabilityFunction + Clone> NormalizedProbabilityMassFunction
    for OldSpParameters<D>
{
}

impl<D: PredictiveProbabilityFunction + Clone> HasPermutation for OldSpParameters<D> {
    fn permutation(&self) -> &Permutation {
        &self.permutation
    }
    fn permutation_mut(&mut self) -> &mut Permutation {
        &mut self.permutation
    }
}

impl<D: PredictiveProbabilityFunction + Clone> HasVectorShrinkage for OldSpParameters<D> {
    fn shrinkage(&self) -> &Shrinkage {
        &self.shrinkage
    }
    fn shrinkage_mut(&mut self) -> &mut Shrinkage {
        &mut self.shrinkage
    }
}

fn engine_full<'a, D: PredictiveProbabilityFunction + Clone, T: Rng>(
    parameters: &'a OldSpParameters<D>,
    target: Option<&[usize]>,
    rng: Option<&mut T>,
) -> (Clustering, f64) {
    engine(
        parameters,
        Clustering::unallocated(parameters.baseline_partition.n_items()),
        vec![vec![0_usize; 0]; parameters.baseline_partition.max_label() + 1],
        target,
        rng,
    )
}

trait EngineFunctions {
    fn multiplier(&self, i: usize) -> f64;
    fn delta(&self, count: usize) -> f64;
}

struct Binder;
impl EngineFunctions for Binder {
    fn multiplier(&self, i: usize) -> f64 {
        2.0 / (((i + 1) * (i + 1)) as f64)
    }
    fn delta(&self, count: usize) -> f64 {
        count as f64
    }
}

struct Vi;
impl EngineFunctions for Vi {
    fn multiplier(&self, i: usize) -> f64 {
        1.0 / ((i + 1) as f64)
    }
    // Since this is a function on integers, we could cache these calculations for more computational efficiency.
    fn delta(&self, count: usize) -> f64 {
        if count == 0 {
            0.0
        } else {
            let n1 = (count + 1) as f64;
            let n0 = count as f64;
            n1 * (n1.log2()) - n0 * (n0.log2())
        }
    }
}

fn engine<'a, D: PredictiveProbabilityFunction + Clone, T: Rng>(
    parameters: &'a OldSpParameters<D>,
    clustering: Clustering,
    counts: Vec<Vec<usize>>,
    target: Option<&[usize]>,
    rng: Option<&mut T>,
) -> (Clustering, f64) {
    match parameters.loss_function {
        LossFunction::BinderDraws(a) => {
            engine_core(Binder, a + 1.0, parameters, clustering, counts, target, rng)
        }
        LossFunction::VI(a) => {
            engine_core(Vi, a + 1.0, parameters, clustering, counts, target, rng)
        }
        _ => panic!("Unsupported loss function."),
    }
}

fn engine_core<'a, D: PredictiveProbabilityFunction + Clone, S: EngineFunctions, T: Rng>(
    functions: S,
    a_plus_one: f64,
    parameters: &'a OldSpParameters<D>,
    mut clustering: Clustering,
    mut counts: Vec<Vec<usize>>,
    target: Option<&[usize]>,
    mut rng: Option<&mut T>,
) -> (Clustering, f64) {
    let mut log_probability = 0.0;
    for i in clustering.n_items_allocated()..clustering.n_items() {
        let item = parameters.permutation.get(i);
        let label_in_baseline = parameters.baseline_partition.get(item);
        let scaled_shrinkage =
            ((i + 1) as f64).powf(parameters.scaling_exponent) * parameters.shrinkage[item];
        let candidate_labels: Vec<usize> = clustering
            .available_labels_for_allocation_with_target(target, item)
            .collect();
        let max_candidate_label = *candidate_labels.iter().max().unwrap();
        if max_candidate_label >= counts[label_in_baseline].len() {
            expand_counts(&mut counts, max_candidate_label + 1)
        }
        let m = functions.multiplier(i);
        let labels_and_log_weights = parameters
            .baseline_ppf
            .log_predictive_weight(item, &candidate_labels, &clustering)
            .into_iter()
            .map(|(label, log_probability)| {
                let nm = clustering.size_of(label);
                let nj = counts[label_in_baseline][label];
                let distance = m * (functions.delta(nm) - a_plus_one * functions.delta(nj));
                (label, log_probability - scaled_shrinkage * distance)
            });
        let (label, log_probability_contribution) = match &mut rng {
            Some(r) => clustering.select(labels_and_log_weights, true, false, 0, Some(r), true),
            None => clustering.select::<Pcg64Mcg, _>(
                labels_and_log_weights,
                true,
                false,
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
    use crate::crp::CrpParameters;
    use crate::prelude::*;

    #[test]
    fn test_goodness_of_fit_constructive() {
        let n_items = 4;
        let discount = 0.1;
        let mass = Mass::new_with_variable_constraint(2.0, discount);
        let discount = Discount::new(discount);
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
            let parameters = OldSpParameters::new(
                target,
                shrinkage,
                permutation,
                baseline_distribution,
                true,
                1.0,
                1.0,
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
            let parameters = OldSpParameters::new(
                target,
                shrinkage,
                permutation,
                baseline_distribution,
                true,
                1.0,
                1.0,
            )
            .unwrap();
            let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
            crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
        }
    }
}
