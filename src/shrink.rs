use crate::prelude::Rate;
use rand::prelude::*;
use rand_distr::{Beta, Distribution};

#[derive(Debug, Clone)]
pub struct Shrinkage(Vec<f64>);

impl Shrinkage {
    pub fn zero(n_items: usize) -> Shrinkage {
        Shrinkage(vec![0.0; n_items])
    }

    pub fn one(n_items: usize) -> Shrinkage {
        Shrinkage(vec![1.0; n_items])
    }

    pub fn constant(value: f64, n_items: usize) -> Option<Shrinkage> {
        if value.is_nan() || value.is_infinite() || value < 0.0 {
            return None;
        }
        Some(Shrinkage(vec![value; n_items]))
    }

    pub fn from_rate(rate: Rate, n_items: usize) -> Shrinkage {
        Shrinkage(vec![rate.unwrap(); n_items])
    }

    pub fn from(w: &[f64]) -> Option<Shrinkage> {
        for ww in w.iter() {
            if ww.is_nan() || ww.is_infinite() || *ww < 0.0 {
                return None;
            }
        }
        Some(Shrinkage(Vec::from(w)))
    }

    pub fn randomize_common<T: Rng>(&mut self, max: f64, shape1: f64, shape2: f64, rng: &mut T) {
        let beta = Beta::new(shape1, shape2).unwrap();
        let value = max * beta.sample(rng);
        for x in &mut self.0 {
            if *x > 0.0 {
                *x = value;
            }
        }
    }

    pub fn randomize_idiosyncratic<T: Rng>(
        &mut self,
        max: f64,
        shape1: f64,
        shape2: f64,
        rng: &mut T,
    ) {
        let beta = Beta::new(shape1, shape2).unwrap();
        for x in &mut self.0 {
            if *x > 0.0 {
                *x = max * beta.sample(rng)
            }
        }
    }

    pub fn n_items(&self) -> usize {
        self.0.len()
    }
}

impl std::ops::Index<usize> for Shrinkage {
    type Output = f64;
    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}
