// Chinese restaurant process

use crate::clust::Clustering;
use crate::mcmc::PriorLogWeight;
use crate::prelude::*;
use crate::prior::{PartitionLogProbability, PartitionSampler};

use rand::Rng;
use statrs::function::gamma::ln_gamma;

#[derive(Debug)]
pub struct CRPParameters {
    mass: Mass,
    discount: Discount,
    n_items: usize,
}

impl CRPParameters {
    pub fn new_with_mass(mass: Mass, n_items: usize) -> Self {
        Self::new_with_mass_and_discount(mass, Discount::new(0.0), n_items)
    }

    pub fn new_with_mass_and_discount(mass: Mass, discount: Discount, n_items: usize) -> Self {
        Self {
            mass,
            discount,
            n_items,
        }
    }
}

impl PriorLogWeight for CRPParameters {
    fn log_weight(&self, item: usize, label: usize, clustering: &Clustering) -> f64 {
        let size = clustering.size_of_without(label, item);
        if size == 0 {
            self.mass.unwrap()
                + (clustering.n_clusters_without(item) as f64) * self.discount.unwrap()
        } else {
            size as f64 - self.discount.unwrap()
        }
        .ln()
    }
}

impl PartitionSampler for CRPParameters {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering {
        let mass = self.mass.unwrap();
        let discount = self.discount.unwrap();
        let mut clustering = Clustering::unallocated(self.n_items);
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
            // We're cheating here a bit in that we know the available labels are sequential from 0 to
            // clustering.n_clusters() + 1, exclusive.  This won't be the case in a Gibbs sampling
            // framework.
            use rand::distributions::{Distribution, WeightedIndex};
            let dist = WeightedIndex::new(weights).unwrap();
            clustering.allocate(i, dist.sample(rng));
        }
        clustering
    }
}

impl PartitionLogProbability for CRPParameters {
    fn log_probability(&self, partition: &Clustering) -> f64 {
        let ni = partition.n_items() as f64;
        let ns = partition.n_clusters() as f64;
        let m = self.mass.unwrap();
        let d = self.discount.unwrap();
        let mut result = ln_gamma(m) - ln_gamma(m + ni);
        if d == 0.0 {
            result += ns * m.ln();
            for label in partition.active_labels() {
                result += ln_gamma(partition.size_of(*label) as f64);
            }
        } else {
            let mut cum_d = 0.0;
            for label in partition.active_labels() {
                result += (m + cum_d).ln();
                cum_d += d;
                let mut cum = 1.0;
                for i in 1..partition.size_of(*label) {
                    cum *= i as f64 - d;
                }
                result += cum.ln();
            }
        }
        result
    }

    fn is_normalized(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clust::Permutation;
    use rand::prelude::*;

    #[test]
    fn test_goodness_of_fit_constructive() {
        let parameters = CRPParameters::new_with_mass(Mass::new(2.0), 5);
        let sample_closure = || parameters.sample(&mut thread_rng());
        let log_prob_closure = |clustering: &mut Clustering| parameters.log_probability(clustering);
        crate::testing::assert_goodness_of_fit(
            100000,
            parameters.n_items,
            sample_closure,
            log_prob_closure,
            1,
            0.001,
        );
    }

    #[test]
    fn test_goodness_of_fit_neal_algorithm3() {
        let parameters =
            CRPParameters::new_with_mass_and_discount(Mass::new(2.0), Discount::new(0.1), 5);
        let l = |_i: usize, _indices: &[usize]| 0.0;
        let mut clustering = Clustering::one_cluster(parameters.n_items);
        let rng = &mut thread_rng();
        let permutation = Permutation::random(clustering.n_items(), rng);
        let sample_closure = || {
            clustering = crate::mcmc::update_neal_algorithm3(
                1,
                &clustering,
                &permutation,
                &parameters,
                &l,
                &mut thread_rng(),
            );
            clustering.relabel(0, None, false).0
        };
        let log_prob_closure = |clustering: &mut Clustering| parameters.log_probability(clustering);
        crate::testing::assert_goodness_of_fit(
            10000,
            parameters.n_items,
            sample_closure,
            log_prob_closure,
            5,
            0.001,
        );
    }

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
    fn test_pmf_without_discount() {
        let parameters = CRPParameters::new_with_mass(Mass::new(1.5), 5);
        let log_prob_closure = |clustering: &mut Clustering| parameters.log_probability(clustering);
        crate::testing::assert_pmf_sums_to_one(parameters.n_items, log_prob_closure, 0.0000001);
    }

    #[test]
    fn test_pmf_with_discount() {
        let parameters =
            CRPParameters::new_with_mass_and_discount(Mass::new(1.5), Discount::new(0.1), 5);
        let log_prob_closure = |clustering: &mut Clustering| parameters.log_probability(clustering);
        crate::testing::assert_pmf_sums_to_one(parameters.n_items, log_prob_closure, 0.0000001);
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__crpparameters_new(
    n_items: i32,
    mass: f64,
    discount: f64,
) -> *mut CRPParameters {
    let d = Discount::new(discount);
    let m = Mass::new_with_variable_constraint(mass, discount);
    // First we create a new object.
    let obj = CRPParameters::new_with_mass_and_discount(m, d, n_items as usize);
    // Then copy it to the heap (so we have a stable pointer to it).
    let boxed_obj = Box::new(obj);
    // Then return a pointer by converting our `Box<_>` into a raw pointer
    Box::into_raw(boxed_obj)
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__crpparameters_free(obj: *mut CRPParameters) {
    // As a rule of thumb, freeing a null pointer is just a noop.
    if obj.is_null() {
        return;
    }
    // Convert the raw pointer back to a Box<_>
    let boxed = Box::from_raw(obj);
    // Then explicitly drop it (optional)
    drop(boxed);
}

/*
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
        ni,
    );
    let matrix: &mut [i32] = std::slice::from_raw_parts_mut(partition_labels_ptr, np * ni);
    let probs: &mut [f64] = std::slice::from_raw_parts_mut(partition_probs_ptr, np);
    if do_sampling != 0 {
        let rng = &mut mk_rng_isaac(seed_ptr);
        for i in 0..np {
            let clustering = parameters.sample(rng);
            let labels = clustering.allocation();
            for j in 0..ni {
                matrix[np * j + i] = (labels[j] + 1) as i32;
            }
            probs[i] = log_pmf(&clustering, &parameters);
        }
    } else {
        for i in 0..np {
            let mut target_labels = Vec::with_capacity(ni);
            for j in 0..ni {
                target_labels.push(matrix[np * j + i] as usize);
            }
            let target = Clustering::from_vector(target_labels);
            probs[i] = log_pmf(&target, &parameters);
        }
    }
}
*/
