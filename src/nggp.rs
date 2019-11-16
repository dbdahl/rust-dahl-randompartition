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

pub fn engine<T: Rng>(
    n_items: usize,
    u: UinNGGP,
    mass: Mass,
    reinforcement: Reinforcement,
    mut target_or_rng: TargetOrRandom<T>,
) -> (Partition, f64) {
    if let TargetOrRandom::Target(t) = &mut target_or_rng {
        assert!(t.is_canonical());
        assert_eq!(t.n_items(), n_items);
    };
    let mut log_probability = 0.0;
    let mut partition = Partition::new(n_items);
    let weight_of_new = mass * (u + 1.0).powf(reinforcement.unwrap());
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
                    subset.n_items() as f64 - reinforcement
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
    u: UinNGGP,
    mass: Mass,
    reinforcement: Reinforcement,
    rng: &mut T,
) -> Partition {
    engine(n_items, u, mass, reinforcement, TargetOrRandom::Random(rng)).0
}

pub fn log_pmf_of_partition_given_u(
    partition: &mut Partition,
    u: UinNGGP,
    mass: Mass,
    reinforcement: Reinforcement,
) -> f64 {
    partition.canonicalize();
    engine::<IsaacRng>(
        partition.n_items(),
        u,
        mass,
        reinforcement,
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

pub fn log_density_of_u(
    u: UinNGGP,
    n_items: usize,
    mass: Mass,
    reinforcement: Reinforcement,
) -> f64 {
    // This should work in theory, but seems to be very unstable in practice.
    let mut partition = Partition::singleton_subsets(n_items);
    log_joint_density(&partition, u, mass, reinforcement)
        - log_pmf_of_partition_given_u(&mut partition, u, mass, reinforcement)
}

pub fn log_joint_density(
    partition: &Partition,
    u: UinNGGP,
    mass: Mass,
    reinforcement: Reinforcement,
) -> f64 {
    let ni = partition.n_items() as f64;
    let ns = partition.n_subsets() as f64;
    let u = u.unwrap();
    let m = mass.unwrap();
    let lm = mass.ln();
    let r = reinforcement.unwrap();
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

pub struct NealParametersNGGP {
    u: UinNGGP,
    mass: Mass,
    reinforcement: Reinforcement,
}

impl NealParametersNGGP {
    pub fn new(u: UinNGGP, mass: Mass, reinforcement: Reinforcement) -> Self {
        Self {
            u,
            mass,
            reinforcement,
        }
    }
}

impl NealFunctions for NealParametersNGGP {
    fn new_weight(&self, _n_subsets: usize) -> f64 {
        self.mass * (self.u + 1.0).powf(self.reinforcement.unwrap())
    }

    fn existing_weight(&self, _n_subsets: usize, n_items: usize) -> f64 {
        (n_items as f64) - self.reinforcement
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quadrature::integrate;

    #[test]
    fn test_sample() {
        let n_partitions = 10000;
        let n_items = 4;
        let mass = Mass::new(2.0);
        let reinforcement = Reinforcement::new(0.0); // Reduces to the DP...
        let u = UinNGGP::new(0.2); // ... where the partition and u are independent.
        let mut samples = PartitionsHolder::with_capacity(n_partitions, n_items);
        let rng = &mut thread_rng();
        for _ in 0..n_partitions {
            samples.push_partition(
                &engine(n_items, u, mass, reinforcement, TargetOrRandom::Random(rng)).0,
            );
        }
        let mut psm = dahl_salso::psm::psm(&samples.view(), true);
        let truth = 1.0 / (1.0 + mass);
        let margin_of_error = 3.58 * (truth * (1.0 - truth) / n_partitions as f64).sqrt();
        assert!(psm.view().data().iter().all(|prob| {
            *prob == 1.0 || (truth - margin_of_error < *prob && *prob < truth + margin_of_error)
        }));
    }

    #[test]
    fn test_joint_density() {
        let n_items = 6;
        let mass = Mass::new(3.0);
        let reinforcement = Reinforcement::new(0.9);
        // u in the outer integral
        let integrand = |u: f64| {
            let u = UinNGGP::new(u);
            Partition::iter(n_items)
                .map(|p| {
                    log_joint_density(&mut Partition::from(&p[..]), u, mass, reinforcement).exp()
                })
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
                    log_joint_density(&mut Partition::from(&p[..]), u, mass, reinforcement).exp()
                };
                integrate(integrand, 0.0, 1000.0, 1e-6).integral
            })
            .sum();
        assert!(0.9999999 <= sum, format!("{}", sum));
        assert!(sum <= 1.0000001, format!("{}", sum));
    }

    #[test]
    fn test_log_pmf() {
        let n_items = 5;
        let u = UinNGGP::new(150.0);
        let mass = Mass::new(2.0);
        let reinforcement = Reinforcement::new(0.7);
        let sum = Partition::iter(n_items)
            .map(|p| {
                log_pmf_of_partition_given_u(&mut Partition::from(&p[..]), u, mass, reinforcement)
                    .exp()
            })
            .sum();
        assert!(0.9999999 <= sum, format!("{}", sum));
        assert!(sum <= 1.0000001, format!("{}", sum));
    }

    #[test]
    fn test_log_density_of_u() {
        let n_items = 5;
        let mass = Mass::new(1.0);
        let reinforcement = Reinforcement::new(0.5);
        let integrand = |u: f64| {
            let u = UinNGGP::new(u);
            log_density_of_u(u, n_items, mass, reinforcement).exp()
        };
        let sum = integrate(integrand, 0.0, 2000.0, 1e-6).integral;
        assert!(0.9999999 <= sum, format!("{}", sum));
        assert!(sum <= 1.0000001, format!("{}", sum));
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__nggp__sample(
    n_partitions: i32,
    n_items: i32,
    u: f64,
    mass: f64,
    reinforcement: f64,
    ptr: *mut i32,
    seed_ptr: *const i32, // Assumed length is 32
) -> () {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let mass = Mass::new(mass);
    let reinforcement = Reinforcement::new(reinforcement);
    let u = UinNGGP::new(u);
    let array: &mut [i32] = slice::from_raw_parts_mut(ptr, np * ni);
    let mut rng = mk_rng_isaac(seed_ptr);
    for i in 0..np {
        let p = engine(ni, u, mass, reinforcement, TargetOrRandom::Random(&mut rng));
        let labels = p.0.labels();
        for j in 0..ni {
            array[np * j + i] = i32::try_from(labels[j].unwrap()).unwrap();
        }
    }
}
