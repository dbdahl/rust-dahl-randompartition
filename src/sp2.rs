// Alternative shrinkage partition distribution

use crate::clust::Clustering;
use crate::distr::{
    FullConditional, PartitionSampler, PredictiveProbabilityFunction, ProbabilityMassFunction,
};
use crate::perm::Permutation;
use crate::shrink::Shrinkage;

use dahl_salso::log2cache::Log2Cache;
use dahl_salso::LossFunction;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;

#[derive(Debug, Clone)]
pub struct Sp2Parameters<D: PredictiveProbabilityFunction + Clone> {
    baseline_partition: Clustering,
    shrinkage: Shrinkage,
    permutation: Permutation,
    baseline_ppf: D,
    loss_function: LossFunction,
    scaling_exponent: f64,
    cache: Log2Cache,
}

impl<D: PredictiveProbabilityFunction + Clone> Sp2Parameters<D> {
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

    pub fn shuffle_permutation<T: Rng>(&mut self, rng: &mut T) {
        self.permutation.shuffle(rng);
    }

    pub fn resample_shrinkage<T: Rng>(&mut self, max: f64, shape1: f64, shape2: f64, rng: &mut T) {
        self.shrinkage =
            Shrinkage::constant_random(self.shrinkage.n_items(), max, shape1, shape2, rng);
    }
}

fn expand_counts(counts: &mut Vec<Vec<usize>>, new_len: usize) {
    counts.iter_mut().map(|x| x.resize(new_len, 0)).collect()
}

impl<D: PredictiveProbabilityFunction + Clone> FullConditional for Sp2Parameters<D> {
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

impl<D: PredictiveProbabilityFunction + Clone> PartitionSampler for Sp2Parameters<D> {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering {
        engine_full(self, None, Some(rng)).0
    }
}

impl<D: PredictiveProbabilityFunction + Clone> ProbabilityMassFunction for Sp2Parameters<D> {
    fn log_pmf(&self, partition: &Clustering) -> f64 {
        engine_full::<D, Pcg64Mcg>(self, Some(partition.allocation()), None).1
    }
    fn is_normalized(&self) -> bool {
        true
    }
}

fn engine_full<'a, D: PredictiveProbabilityFunction + Clone, T: Rng>(
    parameters: &'a Sp2Parameters<D>,
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
    parameters: &'a Sp2Parameters<D>,
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
    _functions: S,
    _a_plus_one: f64,
    parameters: &'a Sp2Parameters<D>,
    mut clustering: Clustering,
    mut counts: Vec<Vec<usize>>,
    target: Option<&[usize]>,
    mut rng: Option<&mut T>,
) -> (Clustering, f64) {
    // In R: Sys.setenv("PUMPKIN_DEBUG"="TRUE")
    let debug = std::env::var("PUMPKIN_DEBUG").unwrap_or_default() == "TRUE";
    if debug {
        println!("#### on alternative shrinkage");
    }
    let mut log_probability = 0.0;
    for i in clustering.n_items_allocated()..clustering.n_items() {
        let item = parameters.permutation.get(i);
        if debug {
            println!("---- i = {}, item = {}", i + 1, item + 1);
        }
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
        let ndenom = counts[label_in_baseline].iter().sum::<usize>() as f64;
        let labels_and_log_weights = parameters
            .baseline_ppf
            .log_predictive_weight(item, &candidate_labels, &clustering)
            .into_iter()
            .map(|(label, log_probability)| {
                let nj = counts[label_in_baseline][label];
                if debug {
                    println!(
                        "label {}: {} {} {} {}",
                        label, log_probability, scaled_shrinkage, nj, ndenom
                    );
                }
                (
                    label,
                    log_probability
                        + if ndenom > 0.0 {
                            scaled_shrinkage * nj as f64 / ndenom
                        } else {
                            0.0
                        },
                )
            });
        if debug {
            let a = labels_and_log_weights
                .clone()
                .map(|x| format!("{} => {}", x.0, x.1.exp()))
                .collect::<Vec<_>>()
                .join(", ");
            println!("{}", a);
        }
        let (label, log_probability_contribution) = match &mut rng {
            Some(r) => clustering.select(labels_and_log_weights, true, 0, Some(r), true),
            None => clustering.select::<Pcg64Mcg, _>(
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
        if debug {
            println!("clustering: {}", clustering)
        }
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
            let parameters = Sp2Parameters::new(
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
            let parameters = Sp2Parameters::new(
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
