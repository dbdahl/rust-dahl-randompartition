use crate::clust::Clustering;
use crate::prelude::*;
use rand::prelude::*;
use rand_distr::{Beta, Distribution};

#[derive(Debug, Clone)]
pub struct Shrinkage(Vec<f64>);

impl Shrinkage {
    pub fn zero(n_items: usize) -> Self {
        Self(vec![0.0; n_items])
    }

    pub fn one(n_items: usize) -> Self {
        Self(vec![1.0; n_items])
    }

    pub fn constant(value: f64, n_items: usize) -> Option<Self> {
        if value.is_nan() || value < 0.0 {
            return None;
        }
        Some(Self(vec![value; n_items]))
    }

    pub fn from(w: &[f64]) -> Option<Self> {
        for ww in w.iter() {
            if ww.is_nan() || *ww < 0.0 {
                return None;
            }
        }
        Some(Self(Vec::from(w)))
    }

    pub fn n_items(&self) -> usize {
        self.0.len()
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.0[..]
    }

    pub fn set_constant(&mut self, new_value: f64) {
        for y in &mut self.0 {
            *y = new_value;
        }
    }

    pub fn rescale_by_reference(&mut self, reference: usize, new_value: f64) {
        let multiplicative_factor = new_value / self.0[reference];
        if (1.0 - multiplicative_factor).abs() > 0.000_000_1 {
            if new_value <= 0.0 {
                panic!("'value' must be nonnegative.");
            }
            for y in &mut self.0 {
                *y *= multiplicative_factor;
            }
        }
    }

    pub fn randomize_common<T: Rng>(
        &mut self,
        max: f64,
        shape1: Shape,
        shape2: Shape,
        rng: &mut T,
    ) {
        let beta = Beta::new(shape1.get(), shape2.get()).unwrap();
        let value = max * beta.sample(rng);
        for x in &mut self.0 {
            if *x > 0.0 {
                *x = value;
            }
        }
    }

    pub fn randomize_common_cluster<T: Rng>(
        &mut self,
        max: f64,
        shape1: Shape,
        shape2: Shape,
        clustering: &Clustering,
        rng: &mut T,
    ) {
        let beta = Beta::new(shape1.get(), shape2.get()).unwrap();
        for k in clustering.available_labels_for_allocation() {
            if clustering.size_of(k) == 0 {
                continue;
            }
            let value = max * beta.sample(rng);
            for i in clustering.items_of(k) {
                if self.0[i] > 0.0 {
                    self.0[i] = value;
                }
            }
        }
    }

    pub fn randomize_idiosyncratic<T: Rng>(
        &mut self,
        max: f64,
        shape1: Shape,
        shape2: Shape,
        rng: &mut T,
    ) {
        let beta = Beta::new(shape1.get(), shape2.get()).unwrap();
        for x in &mut self.0 {
            if *x > 0.0 {
                *x = max * beta.sample(rng)
            }
        }
    }
}

impl std::ops::Index<usize> for Shrinkage {
    type Output = f64;
    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}

fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
