use crate::clust::Clustering;
use crate::prelude::*;
use rand::prelude::*;
use rand_distr::{Distribution, Gamma};

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
        self.0.fill(new_value);
    }

    /// Multiplies every item's shrinkage by the factor that makes the reference
    /// item's `new_value`.
    ///
    /// # Panics
    ///
    /// If the reference item's shrinkage or `new_value` isn't positive and
    /// finite, since the factor would then be 0, infinite, or NaN, and the items'
    /// shrinkages could never be rescaled again.
    pub fn rescale_by_reference(&mut self, reference: usize, new_value: ScalarShrinkage) {
        let old_value = self.0[reference].get();
        let new_value = new_value.get();
        assert!(
            old_value > 0.0 && old_value.is_finite() && new_value > 0.0 && new_value.is_finite(),
            "rescaling the shrinkage from the reference item's {old_value} to {new_value} needs both to be positive and finite"
        );
        let multiplicative_factor = new_value / old_value;
        if (1.0 - multiplicative_factor).abs() > 0.000_000_1 {
            for y in &mut self.0 {
                *y = ScalarShrinkage::new_unchecked(*y * multiplicative_factor);
            }
        }
    }

    pub fn randomize_common<T: Rng>(&mut self, shape: Shape, rate: Rate, rng: &mut T) {
        let gamma = Gamma::new(shape.get(), 1.0 / rate.get()).unwrap();
        let value = ScalarShrinkage::new_unchecked(gamma.sample(rng));
        for x in &mut self.0 {
            if x.get() > 0.0 {
                *x = value;
            }
        }
    }

    pub fn randomize_common_cluster<T: Rng>(
        &mut self,
        shape: Shape,
        rate: Rate,
        clustering: &Clustering,
        rng: &mut T,
    ) {
        let gamma = Gamma::new(shape.get(), 1.0 / rate.get()).unwrap();
        for k in clustering.available_labels_for_allocation() {
            if clustering.size_of(k) == 0 {
                continue;
            }
            let value = ScalarShrinkage::new_unchecked(gamma.sample(rng));
            for i in clustering.items_of(k) {
                if self.0[i].get() > 0.0 {
                    self.0[i] = value;
                }
            }
        }
    }

    pub fn randomize_idiosyncratic<T: Rng>(&mut self, shape: Shape, rate: Rate, rng: &mut T) {
        let gamma = Gamma::new(shape.get(), 1.0 / rate.get()).unwrap();
        for x in &mut self.0 {
            if x.get() > 0.0 {
                *x = ScalarShrinkage::new_unchecked(gamma.sample(rng))
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
