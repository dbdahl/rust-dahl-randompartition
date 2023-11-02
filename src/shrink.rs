use crate::clust::Clustering;
use crate::prelude::*;
use rand::prelude::*;
use rand_distr::{Beta, Distribution};

#[derive(Debug, Clone)]
pub struct Shrinkage(Vec<ScalarShrinkage>);

impl Shrinkage {
    pub fn zero(n_items: usize) -> Self {
        Self(vec![ScalarShrinkage::new_unchecked(0.0); n_items])
    }

    pub fn one(n_items: usize) -> Self {
        Self(vec![ScalarShrinkage::new_unchecked(1.0); n_items])
    }

    pub fn constant(value: ScalarShrinkage, n_items: usize) -> Self {
        Self(vec![value; n_items])
    }

    pub fn from(w: &[f64]) -> Option<Self> {
        let mut vec = Vec::with_capacity(w.len());
        for ww in w.iter() {
            vec.push(ScalarShrinkage::new(*ww)?);
        }
        Some(Self(vec))
    }

    pub fn n_items(&self) -> usize {
        self.0.len()
    }

    pub fn as_slice(&self) -> &[ScalarShrinkage] {
        &self.0[..]
    }

    pub fn set_constant(&mut self, new_value: ScalarShrinkage) {
        for y in &mut self.0 {
            *y = new_value;
        }
    }

    pub fn rescale_by_reference(&mut self, reference: usize, new_value: ScalarShrinkage) {
        let multiplicative_factor = new_value.get() / self.0[reference];
        if (1.0 - multiplicative_factor).abs() > 0.000_000_1 {
            for y in &mut self.0 {
                *y = ScalarShrinkage::new_unchecked(*y * multiplicative_factor);
            }
        }
    }

    pub fn randomize_common<T: Rng>(
        &mut self,
        max: ScalarShrinkage,
        shape1: Shape,
        shape2: Shape,
        rng: &mut T,
    ) {
        let beta = Beta::new(shape1.get(), shape2.get()).unwrap();
        let value = ScalarShrinkage::new_unchecked(max * beta.sample(rng));
        for x in &mut self.0 {
            if x.get() > 0.0 {
                *x = value;
            }
        }
    }

    pub fn randomize_common_cluster<T: Rng>(
        &mut self,
        max: ScalarShrinkage,
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
            let value = ScalarShrinkage::new_unchecked(max * beta.sample(rng));
            for i in clustering.items_of(k) {
                if self.0[i].get() > 0.0 {
                    self.0[i] = value;
                }
            }
        }
    }

    pub fn randomize_idiosyncratic<T: Rng>(
        &mut self,
        max: ScalarShrinkage,
        shape1: Shape,
        shape2: Shape,
        rng: &mut T,
    ) {
        let beta = Beta::new(shape1.get(), shape2.get()).unwrap();
        for x in &mut self.0 {
            if x.get() > 0.0 {
                *x = ScalarShrinkage::new_unchecked(max * beta.sample(rng))
            }
        }
    }
}

impl std::ops::Index<usize> for Shrinkage {
    type Output = ScalarShrinkage;
    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}
