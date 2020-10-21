// Chinese restaurant process

use crate::clust::Clustering;
//use crate::mcmc::PriorLogWeight;
use crate::mcmc::PriorLogWeight2;
use crate::prelude::*;

use dahl_roxido::mk_rng_isaac;
use rand::Rng;
use statrs::function::gamma::ln_gamma;

#[derive(Debug)]
pub struct CRPParameters {
    mass: Mass,
    discount: Discount,
}

impl CRPParameters {
    pub fn new_with_mass(mass: Mass) -> Self {
        Self::new_with_mass_and_discount(mass, Discount::new(0.0))
    }

    pub fn new_with_mass_and_discount(mass: Mass, discount: Discount) -> Self {
        Self { mass, discount }
    }
}

/*
impl PriorLogWeight for CRPParameters {
    fn log_weight(&self, _item_index: usize, subset_index: usize, partition: &Partition) -> f64 {
        let subset = &partition.subsets()[subset_index];
        if subset.n_items() == 0 {
            self.mass.unwrap() + ((partition.n_subsets() - 1) as f64) * self.discount.unwrap()
        } else {
            subset.n_items() as f64 - self.discount.unwrap()
        }
    }
}
*/

impl PriorLogWeight2 for CRPParameters {
    fn log_weight(&self, item: usize, label: usize, clustering: &Clustering) -> f64 {
        let size = clustering.size_of_without(label, item);
        if size == 0 {
            self.mass.unwrap() + (clustering.n_clusters() as f64) * self.discount.unwrap()
        } else {
            size as f64 - self.discount.unwrap()
        }
        .ln()
    }
}

pub fn sample<T: Rng>(n_items: usize, parameters: &CRPParameters, rng: &mut T) -> Clustering {
    let mass = parameters.mass.unwrap();
    let discount = parameters.discount.unwrap();
    let mut clustering = Clustering::unallocated(n_items);
    clustering.allocate(0, 0);
    for i in 1..clustering.n_items() {
        let n_clusters = clustering.n_clusters();
        let weights = clustering.available_labels_for_allocation().map(|label| {
            let n_items_in_cluster = clustering.size_of(label);
            if n_items_in_cluster == 0 {
                mass + (n_clusters as f64) * discount
            } else {
                (n_items_in_cluster as f64) - discount
            }
        });
        use rand::distributions::{Distribution, WeightedIndex};
        let dist = WeightedIndex::new(weights).unwrap();
        clustering.allocate(i, dist.sample(rng));
    }
    clustering
}

pub fn log_pmf(x: &Clustering, parameters: &CRPParameters) -> f64 {
    let ni = x.n_items() as f64;
    let ns = x.n_clusters() as f64;
    let m = parameters.mass.unwrap();
    let lm = m.ln();
    let mut result = ns * lm + ln_gamma(m) - ln_gamma(m + ni);
    for label in x.active_labels() {
        result += ln_gamma(x.size_of(*label) as f64);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    //use crate::mcmc::{update_neal_algorithm3, update_rwmh};
    use rand::prelude::*;

    #[test]
    fn test_goodness_of_fit_constructive() {
        let n_items = 5;
        let parameters = CRPParameters::new_with_mass(Mass::new(2.0));
        let sample_closure = || sample(n_items, &parameters, &mut thread_rng());
        let log_prob_closure = |clustering: &mut Clustering| log_pmf(clustering, &parameters);
        crate::testing::assert_goodness_of_fit(
            100000,
            n_items,
            sample_closure,
            log_prob_closure,
            1,
            0.001,
        );
    }

    /*
    #[test]
    fn test_goodness_of_fit_neal_algorithm3() {
        let n_items = 5;
        let parameters = CRPParameters::new_with_mass(Mass::new(2.0));
        let l = |_i: usize, _indices: &[usize]| 0.0;
        let mut clustering = Partition::one_subset(n_items);
        let permutation = Permutation::natural(clustering.n_items());
        let sample_closure = || {
            clustering = update_neal_algorithm3(
                1,
                &clustering,
                &permutation,
                &parameters,
                &l,
                &mut thread_rng(),
            );
            clustering.clone()
        };
        let log_prob_closure = |clustering: &mut Clustering| log_pmf(clustering, &parameters);
        crate::testing::assert_goodness_of_fit(
            10000,
            n_items,
            sample_closure,
            log_prob_closure,
            5,
            0.001,
        );
    }
    */

    /*
    #[test]
    fn test_goodness_of_fit_rwmh() {
        let n_items = 5;
        let parameters = CRPParameters::new_with_mass(Mass::new(2.0));
        let log_prob_closure = |partition: &mut Partition| log_pmf(partition, &parameters);
        let log_prob_closure2 = |partition: &Partition| log_pmf(partition, &parameters);
        let rate = Rate::new(0.0);
        let discount = 0.1;
        let mass = Mass::new_with_variable_constraint(3.0, discount); // Notice that the mass for the proposal doesn't need to match the prior
        let discount = Discount::new(discount); // Notice that the discount for the proposal doesn't need to match the prior
        let mut p = Partition::one_subset(n_items);
        let mut n_accepts = 0;
        let sample_closure = || {
            let temp = update_rwmh(
                1,
                &p,
                rate,
                mass,
                discount,
                &log_prob_closure2,
                &mut thread_rng(),
            );
            p = temp.0;
            n_accepts += temp.1 as usize;
            p.clone()
        };
        let n_samples = 10000;
        let n_calls_per_sample = 5;
        if let Some(mut string) = crate::testing::assert_goodness_of_fit(
            n_samples,
            n_items,
            sample_closure,
            log_prob_closure,
            n_calls_per_sample,
            0.001,
        ) {
            let x = format!(
                ", acceptance_rate = {:.2}",
                (n_accepts as f64) / (n_calls_per_sample * n_samples) as f64
            );
            string.push_str(&x[..]);
            panic!(string);
        }
    }
    */

    #[test]
    fn test_pmf() {
        let parameters = CRPParameters::new_with_mass(Mass::new(1.5));
        let log_prob_closure = |clustering: &mut Clustering| log_pmf(clustering, &parameters);
        crate::testing::assert_pmf_sums_to_one(5, log_prob_closure, 0.0000001);
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__crp_partition(
    do_sampling: i32,
    n_partitions: i32,
    n_items: i32,
    partition_labels_ptr: *mut i32,
    partition_probs_ptr: *mut f64,
    seed_ptr: *const i32, // Assumed length is 32
    mass: f64,
    discount: f64,
) -> () {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let parameters = CRPParameters::new_with_mass_and_discount(
        Mass::new_with_variable_constraint(mass, discount),
        Discount::new(discount),
    );
    let matrix: &mut [i32] = std::slice::from_raw_parts_mut(partition_labels_ptr, np * ni);
    let probs: &mut [f64] = std::slice::from_raw_parts_mut(partition_probs_ptr, np);
    if do_sampling != 0 {
        let rng = &mut mk_rng_isaac(seed_ptr);
        for i in 0..np {
            let clustering = sample(ni, &parameters, rng);
            let labels = clustering.labels();
            for j in 0..ni {
                matrix[np * j + i] = (labels[j] + 1) as i32;
            }
            probs[i] = log_pmf(&clustering, &parameters);
        }
    } else {
        for i in 0..np {
            let mut target_labels = Vec::with_capacity(ni);
            for j in 0..ni {
                target_labels.push((matrix[np * j + i] - 1) as usize);
            }
            let target = Clustering::from_vector(target_labels);
            probs[i] = log_pmf(&target, &parameters);
        }
    }
}
