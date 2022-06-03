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
    pub fn from_slice<T: Copy + Ord + TryInto<usize>>(x: &[T]) -> Option<Self> {
        let mut y: Vec<usize> = Vec::with_capacity(x.len());
        for &xx in x {
            match xx.try_into() {
                Ok(xxx) => y.push(xxx),
                Err(_) => return None
            }
        }
        let mut copy = y.clone();
        copy.sort_unstable();
        if copy.into_iter().enumerate().any(|(i, x)| x != i) {
            return None
        }
        let n_items = y.len();
        Some(Self {
            x: y,
            n_items,
            natural_and_fixed: false
        })
    }

    pub fn from_vector(x: Vec<usize>) -> Option<Self> {
        let mut y = x.clone();
        y.sort_unstable();
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

    pub fn n_items_before(&self, item: usize) -> usize {
        let mut i = 0;
        loop {
            if self.get(i) == item {
                return i;
            }
            i += 1
        }
    }

    pub fn as_slice(&self) -> &[usize] {
        if self.natural_and_fixed {
            panic!("Not supported.");
        } else {
            &self.x[..]
        }
    }

    pub fn as_slice_until(&self, end: usize) -> &[usize] {
        if self.natural_and_fixed {
            panic!("Not supported.");
        } else {
            &self.x[..end]
        }
    }

    pub fn as_slice_from(&self, start: usize) -> &[usize] {
        if self.natural_and_fixed {
            panic!("Not supported");
        } else {
            &self.x[start..]
        }
    }

}
