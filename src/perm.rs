use rand::prelude::*;

/// A data structure representation a permutation of integers.
///
#[derive(Debug, Clone)]
pub struct Permutation {
    x: Vec<usize>,
    n_items: usize,
    pub natural_and_fixed: bool,
}

impl Permutation {
    pub fn from_slice(x: &[usize]) -> Option<Self> {
        let mut y = Vec::from(x);
        y.sort();
        if y.iter().enumerate().all(|(i, x)| *x == i) {
            Some(Self {
                x: Vec::from(x),
                n_items: y.len(),
                natural_and_fixed: false,
            })
        } else {
            None
        }
    }

    pub fn from_vector(x: Vec<usize>) -> Option<Self> {
        let mut y = x.clone();
        y.sort();
        if y.iter().enumerate().all(|(i, x)| *x == i) {
            Some(Self {
                x,
                n_items: y.len(),
                natural_and_fixed: false,
            })
        } else {
            None
        }
    }

    pub fn natural_and_fixed(n_items: usize) -> Self {
        Self {
            x: Vec::new(),
            n_items,
            natural_and_fixed: true,
        }
    }

    pub fn natural(n_items: usize) -> Self {
        let x = (0..n_items).collect();
        Self {
            x,
            n_items,
            natural_and_fixed: false,
        }
    }

    pub fn random<T: Rng>(n_items: usize, rng: &mut T) -> Self {
        let mut perm = Self::natural(n_items);
        perm.shuffle(rng);
        perm
    }

    pub fn get(&self, i: usize) -> usize {
        if self.natural_and_fixed {
            if i >= self.n_items {
                panic!("Index out of bounds.")
            } else {
                i
            }
        } else {
            self.x[i]
        }
    }

    pub fn shuffle<T: Rng>(&mut self, rng: &mut T) {
        self.x.shuffle(rng)
    }

    pub fn n_items(&self) -> usize {
        self.n_items
    }

    pub fn slice_until(&self, end: usize) -> &[usize] {
        if self.natural_and_fixed {
            panic!("Not supported.");
        } else {
            &self.x[..end]
        }
    }

    pub fn slice_from(&self, start: usize) -> &[usize] {
        if self.natural_and_fixed {
            panic!("Not supported");
        } else {
            &self.x[start..]
        }
    }
}
