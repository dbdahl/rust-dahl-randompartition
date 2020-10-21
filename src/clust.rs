use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::*;
use std::collections::HashMap;
use std::ops::Index;

#[derive(Debug, Clone)]
pub struct Clustering {
    labels: Vec<usize>,
    sizes: Vec<usize>,
    active_labels: Vec<usize>,
    expired_labels: Vec<usize>,
}

impl Index<usize> for Clustering {
    type Output = usize;
    fn index(&self, item: usize) -> &Self::Output {
        &self.labels[item]
    }
}

impl Clustering {
    /// An iterator over all possible partitions for the specified number of items.
    ///
    pub fn iter(n_items: usize) -> ClusteringIterator {
        ClusteringIterator {
            n_items,
            labels: vec![0; n_items],
            max: vec![0; n_items],
            done: false,
            modulus: 1,
        }
    }

    /// A vector of iterators, which together go over all possible partitions for the specified
    /// number of items.  This can be useful in multi-thread situations.
    ///
    pub fn iter_sharded(n_shards: u32, n_items: usize) -> Vec<ClusteringIterator> {
        let mut shards = Vec::with_capacity(n_shards as usize);
        let ns = if n_shards == 0 { 1 } else { n_shards };
        for i in 0..ns {
            let mut iter = ClusteringIterator {
                n_items,
                labels: vec![0; n_items],
                max: vec![0; n_items],
                done: false,
                modulus: ns,
            };
            iter.advance(i);
            shards.push(iter);
        }
        shards
    }

    pub fn unallocated(n_items: usize) -> Self {
        Self {
            labels: vec![usize::max_value(); n_items],
            sizes: {
                let mut sizes = Vec::new();
                sizes.push(0);
                sizes
            },
            active_labels: Vec::new(),
            expired_labels: Vec::new(),
        }
    }

    pub fn one_cluster(n_items: usize) -> Self {
        Self {
            labels: vec![0; n_items],
            sizes: {
                let mut sizes = Vec::new();
                sizes.push(n_items);
                sizes.push(0);
                sizes
            },
            active_labels: {
                let mut active_labels = Vec::new();
                active_labels.push(0);
                active_labels
            },
            expired_labels: Vec::new(),
        }
    }

    pub fn singleton_clusters(n_items: usize) -> Self {
        let labels: Vec<_> = (0..n_items).collect();
        let active_labels = labels.clone();
        Self {
            labels,
            sizes: {
                let mut sizes = Vec::new();
                sizes.resize(n_items, 1);
                sizes.push(0);
                sizes
            },
            active_labels,
            expired_labels: Vec::new(),
        }
    }

    pub fn from_vector(labels: Vec<usize>) -> Self {
        let mut x = Self {
            labels,
            sizes: Vec::new(),
            active_labels: Vec::new(),
            expired_labels: Vec::new(),
        };
        for label in &x.labels {
            if *label >= x.sizes.len() {
                x.sizes.resize(*label + 1, 0)
            }
            x.sizes[*label] += 1;
        }
        for (index, size) in x.sizes.iter().enumerate() {
            if *size != 0 {
                x.active_labels.push(index);
            }
        }
        x.sizes.push(0);
        x
    }

    pub fn from_slice(labels: &[i32]) -> Self {
        Self::from_vector(labels.iter().map(|x| *x as usize).collect())
    }

    pub(crate) unsafe fn push_into_slice_i32(&self, slice: &mut [i32]) {
        slice.iter_mut().zip(self.labels.iter()).for_each(|(x, y)| {
            *x = *y as i32;
        });
    }

    pub fn n_items(&self) -> usize {
        self.labels.len()
    }

    pub fn n_clusters(&self) -> usize {
        self.active_labels.len()
    }

    pub fn labels(&self) -> &Vec<usize> {
        &self.labels
    }

    pub fn size_of(&self, label: usize) -> usize {
        self.sizes[label]
    }

    pub fn size_of_without(&self, label: usize, item: usize) -> usize {
        if self.labels[item] == label {
            self.sizes[label] - 1
        } else {
            self.sizes[label]
        }
    }

    pub fn active_labels(&self) -> &Vec<usize> {
        &self.active_labels
    }

    pub fn new_label(&self) -> usize {
        self.sizes.len() - 1
    }

    pub fn available_labels_for_allocation(&self) -> ClusterLabelsIterator {
        ClusterLabelsIterator {
            iter: self.active_labels.iter(),
            new_label: Some(self.new_label()),
            done: false,
        }
    }

    pub fn available_labels_for_reallocation(&self, item: usize) -> ClusterLabelsIterator {
        let new_label = if self.sizes[self.labels[item]] > 1 {
            Some(self.new_label())
        } else {
            None
        };
        ClusterLabelsIterator {
            iter: self.active_labels.iter(),
            new_label,
            done: false,
        }
    }

    pub fn select_randomly<T: Rng, S: Iterator<Item = (usize, f64)>>(
        &self,
        labels_and_log_weights: S,
        rng: &mut T,
    ) -> usize {
        let (labels, log_weights): (Vec<_>, Vec<_>) = labels_and_log_weights.unzip();
        let max_log_weight = log_weights.iter().cloned().fold(f64::NAN, f64::max);
        let weights = log_weights.iter().map(|x| (*x - max_log_weight).exp());
        let dist = WeightedIndex::new(weights).unwrap();
        labels[dist.sample(rng)]
    }

    pub fn select_randomly_with_weights<T: Rng, S: Iterator<Item = (usize, f64)>>(
        &self,
        labels_and_weights: S,
        rng: &mut T,
    ) -> usize {
        let (labels, weights): (Vec<_>, Vec<_>) = labels_and_weights.unzip();
        let dist = WeightedIndex::new(weights).unwrap();
        labels[dist.sample(rng)]
    }

    pub fn get(&mut self, item: usize) -> usize {
        self.labels[item]
    }

    pub fn allocate(&mut self, item: usize, label: usize) {
        if self.sizes[label] == 0 {
            if label != self.sizes.len() - 1 {
                panic!(
                    "Attempting to allocate using an invalid cluster label.  Use value from 'new_label' function."
                );
            }
            self.active_labels.push(label);
            self.sizes.push(0);
        }
        self.labels[item] = label;
        self.sizes[label] += 1;
    }

    pub fn reallocate(&mut self, item: usize, label: usize) {
        let old_label = self.labels[item];
        if old_label == label {
            return;
        }
        self.allocate(item, label);
        let old_label_size = &mut self.sizes[old_label];
        *old_label_size -= 1;
        if *old_label_size == 0 {
            // Linear search is faster than binary search for integer arrays with less than, say,
            // 150 elements, i.e., active clusters.
            self.active_labels.remove(
                self.active_labels
                    .iter()
                    .position(|x| *x == old_label)
                    .unwrap(),
            );
            self.expired_labels.push(old_label)
        }
    }

    // The functions below are somewhat expensive.

    pub fn items_of(&self, label: usize) -> Vec<usize> {
        let size = self.sizes[label];
        let mut items = Vec::with_capacity(size);
        let mut i = 0;
        while items.len() != size {
            if self.labels[i] == label {
                items.push(i);
            }
            i += 1;
        }
        items
    }

    pub fn standardize(
        &self,
        first_label: usize,
        permutation: Option<&Permutation>,
        with_mapping: bool,
    ) -> (Self, Option<Vec<usize>>) {
        let n_items = self.n_items();
        if let Some(p) = permutation {
            assert_eq!(n_items, p.len());
        };
        let mut labels = Vec::with_capacity(n_items);
        let mut sizes = Vec::new();
        let mut map = HashMap::new();
        let mut next_new_label = first_label;
        for i in 0..n_items {
            let ii = match permutation {
                None => i,
                Some(p) => p.get(i),
            };
            let c = *map.entry(self.labels[ii]).or_insert_with(|| {
                let c = next_new_label;
                next_new_label += 1;
                c
            });
            labels.push(c);
            if c >= sizes.len() {
                sizes.resize(c, 0);
                sizes.push(1);
            } else {
                sizes[c] += 1;
            }
        }
        let mapping = if with_mapping {
            let mut pairs: Vec<_> = map.into_iter().collect();
            pairs.sort_by(|x, y| (*x).1.partial_cmp(&y.1).unwrap());
            Some(pairs.iter().map(|x| x.0).collect())
        } else {
            None
        };
        let active_labels = (first_label..sizes.len()).collect();
        sizes.push(0);
        (
            Self {
                labels,
                sizes,
                active_labels,
                expired_labels: Vec::new(),
            },
            mapping,
        )
    }
}

#[doc(hidden)]
pub struct ClusteringIterator {
    n_items: usize,
    labels: Vec<usize>,
    max: Vec<usize>,
    done: bool,
    modulus: u32,
}

impl ClusteringIterator {
    fn advance(&mut self, times: u32) {
        for _ in 0..times {
            let mut i = self.n_items - 1;
            while (i > 0) && (self.labels[i] == self.max[i - 1] + 1) {
                self.labels[i] = 0;
                self.max[i] = self.max[i - 1];
                i -= 1;
            }
            if i == 0 {
                self.done = true;
                return;
            }
            self.labels[i] += 1;
            let m = self.max[i].max(self.labels[i]);
            self.max[i] = m;
            i += 1;
            while i < self.n_items {
                self.max[i] = m;
                self.labels[i] = 0;
                i += 1;
            }
        }
    }
}

impl Iterator for ClusteringIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            None
        } else {
            let result = Some(self.labels.clone());
            self.advance(self.modulus);
            result
        }
    }
}

pub struct ClusterLabelsIterator<'a> {
    iter: std::slice::Iter<'a, usize>,
    new_label: Option<usize>,
    done: bool,
}

impl<'a> Iterator for ClusterLabelsIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.done {
            match self.iter.next() {
                Some(x) => Some(*x),
                None => {
                    self.done = true;
                    self.new_label
                }
            }
        } else {
            None
        }
    }
}

/// A data structure representation a permutation of integers.
///
#[derive(Debug)]
pub struct Permutation {
    x: Vec<usize>,
    n_items: usize,
    natural: bool,
}

impl Permutation {
    pub fn from_slice(x: &[usize]) -> Option<Self> {
        let mut y = Vec::from(x);
        y.sort();
        if y.iter().enumerate().all(|(i, x)| *x == i) {
            Some(Self {
                x: Vec::from(x),
                n_items: y.len(),
                natural: false,
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
                natural: false,
            })
        } else {
            None
        }
    }

    pub fn natural(n_items: usize) -> Self {
        Self {
            x: Vec::new(),
            n_items,
            natural: true,
        }
    }

    pub fn random<T: Rng>(n_items: usize, rng: &mut T) -> Self {
        let mut perm = Self::natural(n_items);
        perm.shuffle(rng);
        perm
    }

    pub fn get(&self, i: usize) -> usize {
        if self.natural {
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

    pub fn len(&self) -> usize {
        self.n_items
    }

    pub fn slice_until(&self, end: usize) -> &[usize] {
        if self.natural {
            panic!("Not supported.");
        } else {
            &self.x[..end]
        }
    }

    pub fn slice_from(&self, start: usize) -> &[usize] {
        if self.natural {
            panic!("Not supported");
        } else {
            &self.x[start..]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::{Debug, Write};

    fn check_output<T: Debug>(clustering: &T, expected_output: &str) {
        let mut output = String::new();
        write!(&mut output, "{:?}", clustering).expect("Oops");
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_initialization() {
        let clustering = Clustering::one_cluster(5);
        check_output(
            &clustering,
            "Clustering { labels: [0, 0, 0, 0, 0], sizes: [5, 0], active_labels: [0], expired_labels: [] }",
        );
        let clustering = Clustering::singleton_clusters(5);
        check_output(
            &clustering,
            "Clustering { labels: [0, 1, 2, 3, 4], sizes: [1, 1, 1, 1, 1, 0], active_labels: [0, 1, 2, 3, 4], expired_labels: [] }",
        );
        let clustering = Clustering::from_vector(vec![2, 2, 4, 3, 4]);
        check_output(
            &clustering,
            "Clustering { labels: [2, 2, 4, 3, 4], sizes: [0, 0, 2, 1, 2, 0], active_labels: [2, 3, 4], expired_labels: [] }",
        );
        let (clustering, map) = clustering.standardize(1, None, true);
        check_output(
            &clustering,
            "Clustering { labels: [1, 1, 2, 3, 2], sizes: [0, 2, 2, 1, 0], active_labels: [1, 2, 3], expired_labels: [] }",
        );
        check_output(&map, "Some([2, 4, 3])");
    }

    #[test]
    #[should_panic]
    fn test_add_to_nonexisting_cluster() {
        let mut clustering = Clustering::singleton_clusters(5);
        let new_label = 6;
        clustering.reallocate(1, new_label);
    }

    #[test]
    fn test_add_to_existing_cluster() {
        let mut clustering = Clustering::singleton_clusters(5);
        let new_label = clustering.new_label();
        clustering.reallocate(1, new_label);
        check_output(&clustering, "Clustering { labels: [0, 5, 2, 3, 4], sizes: [1, 0, 1, 1, 1, 1, 0], active_labels: [0, 2, 3, 4, 5], expired_labels: [1] }");
        clustering.new_label();
        check_output(&clustering, "Clustering { labels: [0, 5, 2, 3, 4], sizes: [1, 0, 1, 1, 1, 1, 0], active_labels: [0, 2, 3, 4, 5], expired_labels: [1] }");
        clustering.reallocate(3, 2);
    }

    #[test]
    #[should_panic]
    fn test_add_to_improper_cluster() {
        let mut clustering = Clustering::from_vector(vec![0, 5, 2, 2, 4]);
        clustering.new_label();
        clustering.reallocate(3, 1);
    }

    #[test]
    fn test_items_of() {
        let clustering = Clustering::from_vector(vec![2, 2, 4, 3, 4]);
        check_output(&clustering.items_of(4), "[2, 4]");
    }

    #[test]
    fn test_unique_labels() {
        let mut clustering = Clustering::from_vector(vec![2, 2, 4, 3, 4]);
        check_output(&clustering.active_labels(), "[2, 3, 4]");
        clustering.reallocate(3, 2);
        clustering.reallocate(1, 5);
        check_output(&clustering.active_labels(), "[2, 4, 5]");
    }
}
