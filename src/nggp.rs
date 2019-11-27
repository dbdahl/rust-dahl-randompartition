// Normalized generalized gamma process

use crate::mcmc::NealFunctions;
use crate::prelude::*;
use crate::TargetOrRandom;

use dahl_partition::*;
use dahl_roxido::mk_rng_isaac;
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::*;
use rand_distr::Normal;
use rand_isaac::IsaacRng;
use statrs::function::gamma::ln_gamma;
use std::convert::TryFrom;
use std::slice;

pub struct NGGPParameters {
    u: UinNGGP,
    mass: Mass,
    reinforcement: Reinforcement,
}

impl NGGPParameters {
    pub fn new(u: UinNGGP, mass: Mass, reinforcement: Reinforcement) -> Self {
        Self {
            u,
            mass,
            reinforcement,
        }
    }
}

impl NealFunctions for NGGPParameters {
    fn new_weight(&self, _n_subsets: usize) -> f64 {
        self.mass * (self.u + 1.0).powf(self.reinforcement.unwrap())
    }

    fn existing_weight(&self, _n_subsets: usize, n_items: usize) -> f64 {
        n_items as f64 - self.reinforcement
    }
}

pub fn engine<T: Rng>(
    n_items: usize,
    parameters: &NGGPParameters,
    mut target_or_rng: TargetOrRandom<T>,
) -> (Partition, f64) {
    if let TargetOrRandom::Target(t) = &mut target_or_rng {
        assert_eq!(t.n_items(), n_items);
        t.canonicalize();
    };
    let mut log_probability = 0.0;
    let mut partition = Partition::new(n_items);
    let weight_of_new =
        parameters.mass * (parameters.u + 1.0).powf(parameters.reinforcement.unwrap());
    for i in 0..partition.n_items() {
        // Ensure there is an empty subset
        match partition.subsets().last() {
            None => partition.new_subset(),
            Some(last) => {
                if !last.is_empty() {
                    partition.new_subset()
                }
            }
        }
        let probs: Vec<(usize, f64)> = partition
            .subsets()
            .iter()
            .map(|subset| {
                if subset.is_empty() {
                    weight_of_new
                } else {
                    subset.n_items() as f64 - parameters.reinforcement
                }
            })
            .enumerate()
            .collect();
        let subset_index = match &mut target_or_rng {
            TargetOrRandom::Random(rng) => {
                let dist = WeightedIndex::new(probs.iter().map(|x| x.1)).unwrap();
                dist.sample(*rng)
            }
            TargetOrRandom::Target(t) => t.label_of(i).unwrap(),
        };
        let numerator = probs[subset_index].1;
        let denominator = probs.iter().fold(0.0, |sum, x| sum + x.1);
        log_probability += (numerator / denominator).ln();
        partition.add_with_index(i, subset_index);
    }
    partition.canonicalize();
    (partition, log_probability)
}

pub fn sample_partition_given_u<T: Rng>(
    n_items: usize,
    parameters: &NGGPParameters,
    rng: &mut T,
) -> Partition {
    engine(n_items, parameters, TargetOrRandom::Random(rng)).0
}

pub fn log_pmf_of_partition_given_u(partition: &mut Partition, parameters: &NGGPParameters) -> f64 {
    partition.canonicalize();
    engine::<IsaacRng>(
        partition.n_items(),
        parameters,
        TargetOrRandom::Target(partition),
    )
    .1
}

pub fn log_full_conditional_of_log_u(
    // Up to the normalizing constant
    v: f64,
    partition: &Partition,
    mass: Mass,
    reinforcement: Reinforcement,
) -> f64 {
    let u = v.exp();
    let ni = partition.n_items() as f64;
    let ns = partition.n_subsets() as f64;
    let m = mass.unwrap();
    let r = reinforcement.unwrap();
    (v * ni) - (ni - r * ns) * (u + 1.0).ln() - (m / r) * ((u + 1.0).powf(r) - 1.0)
}

pub fn update_u<T: Rng>(
    u: UinNGGP,
    partition: &Partition,
    mass: Mass,
    reinforcement: Reinforcement,
    n_updates: u32,
    rng: &mut T,
) -> UinNGGP {
    let mut current = u.ln();
    let mut f_current = log_full_conditional_of_log_u(current, partition, mass, reinforcement);
    let normal = Normal::new(0.0, 0.5_f64.sqrt()).unwrap();
    for _ in 0..n_updates {
        let proposal = current + normal.sample(rng);
        let f_proposal = log_full_conditional_of_log_u(proposal, partition, mass, reinforcement);
        if rng.gen_range(0.0_f64, 1.0_f64).ln() < f_proposal - f_current {
            current = proposal;
            f_current = f_proposal;
        }
    }
    UinNGGP::new(current.exp())
}

pub fn log_density_of_u(n_items: usize, parameters: &NGGPParameters) -> f64 {
    // This should work in theory, but seems to be very unstable in practice.
    let mut partition = Partition::singleton_subsets(n_items);
    let lower = log_joint_density(&partition, parameters)
        - log_pmf_of_partition_given_u(&mut partition, parameters);
    let mut partition = Partition::one_subset(n_items);
    let upper = log_joint_density(&partition, parameters)
        - log_pmf_of_partition_given_u(&mut partition, parameters);
    (lower + upper) / 2.0
}

pub fn log_joint_density(partition: &Partition, parameters: &NGGPParameters) -> f64 {
    let ni = partition.n_items() as f64;
    let ns = partition.n_subsets() as f64;
    let u = parameters.u.unwrap();
    let m = parameters.mass.unwrap();
    let lm = m.ln();
    let r = parameters.reinforcement.unwrap();
    let mut result = ns * lm + (ni - 1.0) * u.ln()
        - ln_gamma(ni)
        - (ni - r * ns) * (u + 1.0).ln()
        - (m / r) * ((u + 1.0).powf(r) - 1.0);
    for subset in partition.subsets() {
        result += ln_gamma(subset.n_items() as f64 - r);
    }
    result -= ns * ln_gamma(1.0 - r);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcmc::{update_neal_algorithm3, update_rwmh};
    use quadrature::integrate;

    #[test]
    fn test_goodness_of_fit_constructive() {
        let n_items = 5;
        let parameters =
            NGGPParameters::new(UinNGGP::new(300.0), Mass::new(2.0), Reinforcement::new(0.2));
        let sample_closure = || sample_partition_given_u(n_items, &parameters, &mut thread_rng());
        let log_prob_closure =
            |partition: &mut Partition| log_pmf_of_partition_given_u(partition, &parameters);
        if let Some(string) = crate::testing::assert_goodness_of_fit(
            10000,
            n_items,
            sample_closure,
            log_prob_closure,
            1,
            0.001,
        ) {
            panic!("{}", string);
        }
    }

    #[test]
    fn test_goodness_of_fit_neal_algorithm3() {
        let n_items = 5;
        let parameters =
            NGGPParameters::new(UinNGGP::new(300.0), Mass::new(2.0), Reinforcement::new(0.2));
        let l = |_i: usize, _indices: &[usize]| 0.0;
        let mut p = Partition::one_subset(n_items);
        let sample_closure = || {
            p = update_neal_algorithm3(1, &p, &parameters, &l, &mut thread_rng());
            p.clone()
        };
        let log_prob_closure =
            |partition: &mut Partition| log_pmf_of_partition_given_u(partition, &parameters);
        if let Some(string) = crate::testing::assert_goodness_of_fit(
            10000,
            n_items,
            sample_closure,
            log_prob_closure,
            3,
            0.001,
        ) {
            panic!("{}", string);
        }
    }

    #[test]
    fn test_goodness_of_fit_rwmh() {
        let n_items = 5;
        let parameters =
            NGGPParameters::new(UinNGGP::new(1.0), Mass::new(1.0), Reinforcement::new(0.1));
        let log_prob_closure =
            |partition: &mut Partition| log_pmf_of_partition_given_u(partition, &parameters);
        let log_prob_closure2 = |partition: &Partition| {
            let mut p = partition.clone();
            p.canonicalize();
            log_pmf_of_partition_given_u(&mut p, &parameters)
        };
        let rate = Rate::new(1.0);
        let mass = Mass::new(1.5); // Notice that the mass for the proposal doesn't need to match the prior
        let mut p = Partition::one_subset(n_items);
        let sample_closure = || {
            p = update_rwmh(1, &p, rate, mass, &log_prob_closure2, &mut thread_rng()).0;
            p.clone()
        };
        if let Some(string) = crate::testing::assert_goodness_of_fit(
            10000,
            n_items,
            sample_closure,
            log_prob_closure,
            5,
            0.001,
        ) {
            panic!("{}", string);
        }
    }

    #[test]
    fn test_pmf() {
        let parameters =
            NGGPParameters::new(UinNGGP::new(400.0), Mass::new(2.0), Reinforcement::new(0.1));
        let log_prob_closure =
            |partition: &mut Partition| log_pmf_of_partition_given_u(partition, &parameters);
        crate::testing::assert_pmf_sums_to_one(5, log_prob_closure, 0.0000001);
    }

    #[test]
    fn test_joint_density() {
        let n_items = 6;
        let mass = Mass::new(3.0);
        let reinforcement = Reinforcement::new(0.9);
        // u in the outer integral
        let integrand = |u: f64| {
            let u = UinNGGP::new(u);
            let parameters = NGGPParameters::new(u, mass, reinforcement);
            Partition::iter(n_items)
                .map(|p| log_joint_density(&mut Partition::from(&p[..]), &parameters).exp())
                .sum()
        };
        let sum = integrate(integrand, 0.0, 1000.0, 1e-6).integral;
        assert!(0.9999999 <= sum, format!("{}", sum));
        assert!(sum <= 1.0000001, format!("{}", sum));
        // u in the inner integral
        let sum = Partition::iter(n_items)
            .map(|p| {
                let integrand = |u: f64| {
                    let u = UinNGGP::new(u);
                    let parameters = NGGPParameters::new(u, mass, reinforcement);
                    log_joint_density(&mut Partition::from(&p[..]), &parameters).exp()
                };
                integrate(integrand, 0.0, 1000.0, 1e-6).integral
            })
            .sum();
        assert!(0.9999999 <= sum, format!("{}", sum));
        assert!(sum <= 1.0000001, format!("{}", sum));
    }

    #[test]
    fn test_log_density_of_u() {
        let n_items = 4;
        let mass = Mass::new(2.0);
        let reinforcement = Reinforcement::new(0.1);
        let integrand = |u: f64| {
            let u = UinNGGP::new(u);
            let parameters = NGGPParameters::new(u, mass, reinforcement);
            log_density_of_u(n_items, &parameters).exp()
        };
        let sum = integrate(integrand, 0.0, 2000.0, 1e-6).integral;
        let epsilon = 0.011;
        assert!(
            1.0 - epsilon <= sum && sum <= 1.0 + epsilon,
            format!("Total probability should be one, but is {}.", sum)
        );
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__nggp_partition(
    do_sampling: i32,
    n_partitions: i32,
    n_items: i32,
    partition_labels_ptr: *mut i32,
    partition_probs_ptr: *mut f64,
    seed_ptr: *const i32, // Assumed length is 32
    u: f64,
    mass: f64,
    reinforcement: f64,
    n_updates_for_u: i32,
) -> () {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let mut u = UinNGGP::new(u);
    let mass = Mass::new(mass);
    let reinforcement = Reinforcement::new(reinforcement);
    let matrix: &mut [i32] = slice::from_raw_parts_mut(partition_labels_ptr, np * ni);
    let probs: &mut [f64] = slice::from_raw_parts_mut(partition_probs_ptr, np);
    if do_sampling != 0 {
        let mut rng = mk_rng_isaac(seed_ptr);
        // let mut p = Partition::one_subset(ni);
        // let l = |_i: usize, _indices: &[usize]| 0.0;
        for i in 0..np {
            let parameters = NGGPParameters::new(u, mass, reinforcement);
            let p = engine(ni, &parameters, TargetOrRandom::Random(&mut rng));
            // p = crate::mcmc::update_neal_algorithm3(1, &p, &parameters, &l, &mut rng);
            let labels = p.0.labels();
            // let labels = p.labels();
            for j in 0..ni {
                matrix[np * j + i] = i32::try_from(labels[j].unwrap()).unwrap();
            }
            probs[i] = p.1;
            // probs[i] = 0.0;
            u = super::nggp::update_u(
                u,
                &p.0,
                // &p,
                mass,
                reinforcement,
                n_updates_for_u as u32,
                &mut rng,
            );
        }
    } else {
        for i in 0..np {
            let mut target_labels = Vec::with_capacity(ni);
            for j in 0..ni {
                target_labels.push(matrix[np * j + i]);
            }
            let mut target = Partition::from(&target_labels[..]);
            let parameters = NGGPParameters::new(u, mass, reinforcement);
            let p = engine::<IsaacRng>(ni, &parameters, TargetOrRandom::Target(&mut target));
            probs[i] = p.1;
        }
    }
}
