use rand::prelude::*;

/// A data structure representation a permutation of integers.
///
#[derive(Debug, Clone)]
pub struct Permutation {
    x: Vec<usize>,
    n_items: usize,
    pub natural_and_fixed: bool,
    indices_for_partial_shuffle: Vec<usize>,
    flag_for_partial_shuffle: bool,
}

impl Permutation {
    pub fn from_slice<T: Copy + Ord + TryInto<usize>>(x: &[T]) -> Option<Self> {
        let mut y: Vec<usize> = Vec::with_capacity(x.len());
        for &xx in x {
            match xx.try_into() {
                Ok(xxx) => y.push(xxx),
                Err(_) => return None,
            }
        }
        let mut copy = y.clone();
        copy.sort_unstable();
        if copy.iter().enumerate().any(|(i, &x)| x != i) {
            return None;
        }
        let n_items = y.len();
        Some(Self {
            x: y,
            n_items,
            natural_and_fixed: false,
            indices_for_partial_shuffle: copy,
            flag_for_partial_shuffle: true,
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
                indices_for_partial_shuffle: y,
                flag_for_partial_shuffle: true,
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
            indices_for_partial_shuffle: Vec::new(),
            flag_for_partial_shuffle: true,
        }
    }

    pub fn natural(n_items: usize) -> Self {
        let x: Vec<_> = (0..n_items).collect();
        Self {
            x: x.clone(),
            n_items,
            natural_and_fixed: false,
            indices_for_partial_shuffle: x,
            flag_for_partial_shuffle: true,
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
        if self.natural_and_fixed {
            return;
        }
        self.x.shuffle(rng)
    }

    pub fn partial_shuffle<T: Rng>(&mut self, amount: usize, rng: &mut T) {
        if self.natural_and_fixed {
            return;
        }
        if amount == 0 {
            return;
        }
        let (y, z) = self
            .indices_for_partial_shuffle
            .partial_shuffle(rng, amount);
        let tmp = self.x[y[0]];
        for i in 0..(amount - 1) {
            self.x[y[i]] = self.x[y[i + 1]]
        }
        self.x[y[amount - 1]] = tmp;
        self.flag_for_partial_shuffle = y.as_ptr() < z.as_ptr();
    }

    pub fn partial_shuffle_undo(&mut self, amount: usize) {
        if amount <= 1 {
            return;
        }
        let y = if self.flag_for_partial_shuffle {
            &self.indices_for_partial_shuffle[0..amount]
        } else {
            &self.indices_for_partial_shuffle[(self.n_items - amount)..]
        };
        let tmp = self.x[y[amount - 1]];
        for i in (0..(amount - 1)).rev() {
            self.x[y[i + 1]] = self.x[y[i]]
        }
        self.x[y[0]] = tmp;
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

#[cfg(test)]
mod tests {
    use crate::perm::Permutation;
    use rand::{thread_rng, Rng};

    #[test]
    fn partial_shuffle() {
        let rng = &mut thread_rng();
        for _ in 0..500 {
            let n_items = rng.gen_range(1..100_usize);
            let k = rng.gen_range(0..n_items);
            let mut perm = Permutation::random(n_items, rng);
            perm.partial_shuffle(k, rng);
            let perm2 = Permutation::from_vector(perm.x);
            assert!(perm2.is_some());
        }
    }

    #[test]
    fn partial_shuffle_undo() {
        let rng = &mut thread_rng();
        for _ in 0..500 {
            let n_items = rng.gen_range(1..100_usize);
            let k = rng.gen_range(0..n_items);
            let mut perm = Permutation::random(n_items, rng);
            let original = perm.clone();
            perm.partial_shuffle(k, rng);
            perm.partial_shuffle_undo(k);
            for (i, j) in perm.x.iter().zip(original.x.iter()) {
                assert_eq!(i, j);
            }
        }
    }
}
