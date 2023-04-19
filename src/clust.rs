use crate::perm::Permutation;

use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::*;
use std::collections::HashMap;
use std::fmt::Display;

#[derive(Debug, Clone)]
pub struct Clustering {
    allocation: Vec<usize>,
    sizes: Vec<usize>,
    active_labels: Vec<usize>,
    available_labels: Vec<usize>,
}

impl Display for Clustering {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let p = self
            .allocation
            .iter()
            .map(|x| {
                if *x == Clustering::UNALLOCATED {
                    "_".to_string()
                } else {
                    x.to_string()
                }
            })
            .collect::<Vec<String>>()
            .concat();
        write!(f, "{:?}", p)
    }
}

impl Clustering {
    const UNALLOCATED: usize = usize::MAX;

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
            allocation: vec![Clustering::UNALLOCATED; n_items],
            sizes: Vec::new(),
            active_labels: Vec::new(),
            available_labels: Vec::new(),
        }
    }

    pub fn one_cluster(n_items: usize) -> Self {
        Self {
            allocation: vec![0; n_items],
            sizes: vec![n_items; 1],
            active_labels: vec![0; 1],
            available_labels: Vec::new(),
        }
    }

    pub fn singleton_clusters(n_items: usize) -> Self {
        let allocation: Vec<_> = (0..n_items).collect();
        let active_labels = allocation.clone();
        Self {
            allocation,
            sizes: vec![1; n_items],
            active_labels,
            available_labels: Vec::new(),
        }
    }

    pub fn from_vector(labels: Vec<usize>) -> Self {
        let mut x = Self {
            allocation: labels,
            sizes: Vec::new(),
            active_labels: Vec::new(),
            available_labels: Vec::new(),
        };
        for label in &x.allocation {
            if *label >= x.sizes.len() {
                x.sizes.resize(*label + 1, 0)
            }
            x.sizes[*label] += 1;
        }
        for (index, size) in x.sizes.iter().enumerate() {
            if *size == 0 {
                x.available_labels.push(index);
            } else {
                x.active_labels.push(index);
            }
        }
        x
    }

    pub fn from_slice(labels: &[i32]) -> Self {
        Self::from_vector(labels.iter().map(|x| *x as usize).collect())
    }

    pub fn exclude_label(&mut self, label: usize) {
        if label >= self.sizes.len() {
            panic!("Cluster with label {} does not already exists.", label);
        }
        if self.sizes[label] != 0 {
            panic!("Cluster with label {} is not empty.", label);
        }
        self.available_labels.swap_remove(
            self.available_labels
                .iter()
                .rposition(|x| *x == label)
                .unwrap(),
        );
    }

    pub fn n_items(&self) -> usize {
        self.allocation.len()
    }

    pub fn n_items_allocated(&self) -> usize {
        self.sizes.iter().sum()
    }

    pub fn n_items_allocated_without(&self, item: usize) -> usize {
        self.n_items_allocated()
            - (if self.allocation[item] == Clustering::UNALLOCATED {
                0
            } else {
                1
            })
    }

    pub fn n_clusters(&self) -> usize {
        self.active_labels.len()
    }

    pub fn n_clusters_without(&self, item: usize) -> usize {
        let label = self.allocation[item];
        if label == Clustering::UNALLOCATED || self.size_of(label) > 1 {
            self.active_labels.len()
        } else {
            self.active_labels.len() - 1
        }
    }

    pub fn max_label(&self) -> usize {
        self.sizes.len() - 1
    }

    pub fn allocation(&self) -> &Vec<usize> {
        &self.allocation
    }

    pub fn into_vector(self) -> Vec<usize> {
        self.allocation
    }

    pub fn into_slice<T: FromUsize>(self, slice: &mut [T]) {
        for (old_label, slice_item) in self.allocation.iter().zip(slice.iter_mut()) {
            *slice_item = T::from_usize(*old_label);
        }
    }

    pub fn size_of(&self, label: usize) -> usize {
        match self.sizes.get(label) {
            Some(size) => *size,
            None => 0,
        }
    }

    pub fn size_of_without(&self, label: usize, item: usize) -> usize {
        if self.allocation[item] == label {
            self.size_of(label) - 1
        } else {
            self.size_of(label)
        }
    }

    pub fn active_labels(&self) -> &Vec<usize> {
        &self.active_labels
    }

    pub fn new_label(&self) -> usize {
        match self.available_labels.last() {
            Some(label) => *label,
            None => self.sizes.len(),
        }
    }

    pub fn available_labels_for_allocation(&self) -> std::ops::RangeInclusive<usize> {
        0..=self.active_labels.len()
    }

    pub fn available_labels_for_allocation_with_target(
        &self,
        target: Option<&[usize]>,
        item: usize,
    ) -> ClusterLabelsIterator {
        let new_label = match target {
            Some(target) => {
                let what = target[item];
                if self.active_labels.contains(&what) {
                    self.new_label()
                } else {
                    what
                }
            }
            None => self.new_label(),
        };
        ClusterLabelsIterator {
            iter: self.active_labels.iter(),
            new_label: Some(new_label),
            done: false,
        }
    }

    pub fn available_labels_for_reallocation(&self, item: usize) -> ClusterLabelsIterator {
        let new_label = if self.size_of(self.allocation[item]) > 1 {
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

    pub fn select<T: Rng, S: Iterator<Item = (usize, f64)>>(
        &self,
        labels_and_weights: S,
        weights_on_log_scale: bool,
        weights_are_probabilities: bool,
        label: usize,
        rng: Option<&mut T>,
        with_log_probability: bool,
    ) -> (usize, f64) {
        let (labels, weights): (Vec<_>, Vec<_>) = labels_and_weights.unzip();
        let w = if weights_on_log_scale {
            let max_log_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            weights
                .iter()
                .map(|x| (*x - max_log_weight).exp())
                .collect::<Vec<_>>()
        } else {
            weights
        };
        let (label, index) = match rng {
            Some(r) => {
                let index = if w.len() == 1 {
                    0
                } else {
                    WeightedIndex::new(w.iter()).unwrap().sample(r)
                };
                (labels[index], index)
            }
            None => {
                if with_log_probability {
                    (label, labels.iter().position(|x| *x == label).unwrap())
                } else {
                    (label, 0)
                }
            }
        };
        if with_log_probability {
            let ww = w[index];
            if weights_are_probabilities {
                (label, ww.ln())
            } else {
                (label, (ww / w.iter().sum::<f64>()).ln())
            }
        } else {
            (label, 0.0)
        }
    }

    pub fn get(&self, item: usize) -> usize {
        let x = self.allocation[item];
        if x == Clustering::UNALLOCATED {
            panic!("Item is not allocated.");
        }
        x
    }

    pub fn allocate(&mut self, item: usize, label: usize) {
        let old_label = self.allocation[item];
        if old_label == label {
            return;
        }
        self.allocation[item] = label;
        if label != Clustering::UNALLOCATED {
            if label >= self.sizes.len() {
                if label > self.sizes.len() {
                    self.available_labels.reserve(label - self.sizes.len());
                    self.available_labels.extend(self.sizes.len()..label);
                }
                self.sizes.resize(label + 1, 0);
                self.active_labels.push(label);
            } else if self.sizes[label] == 0 {
                self.available_labels.swap_remove(
                    self.available_labels
                        .iter()
                        .rposition(|x| *x == label)
                        .unwrap(),
                );
                self.active_labels.push(label);
            }
            self.sizes[label] += 1;
        }
        if old_label != Clustering::UNALLOCATED {
            let old_label_size = &mut self.sizes[old_label];
            *old_label_size -= 1;
            if *old_label_size == 0 {
                // Linear search is faster than binary search for integer arrays with less than, say,
                // 150 elements, i.e., active clusters.
                self.active_labels.swap_remove(
                    self.active_labels
                        .iter()
                        .rposition(|x| *x == old_label)
                        .unwrap(),
                );
                self.available_labels.push(old_label);
            }
        }
    }

    pub fn remove(&mut self, item: usize) {
        self.allocate(item, Clustering::UNALLOCATED);
    }

    // The functions below are somewhat expensive.

    pub fn items_of(&self, label: usize) -> Vec<usize> {
        let size = self.size_of(label);
        let mut items = Vec::with_capacity(size);
        if size > 0 {
            for (i, k) in self.allocation.iter().enumerate() {
                if *k == label {
                    items.push(i);
                    if items.len() == size {
                        break;
                    }
                }
            }
        }
        items
    }

    pub fn items_of_without(&self, label: usize, item: usize) -> Vec<usize> {
        let size = self.size_of_without(label, item);
        let mut items = Vec::with_capacity(size);
        if size > 0 {
            for (i, k) in self.allocation.iter().enumerate() {
                if i != item && *k == label {
                    items.push(i);
                    if items.len() == size {
                        break;
                    }
                }
            }
        }
        items
    }

    pub fn standardize(&self) -> Self {
        self.relabel(0, None, false).0
    }

    pub fn standardize_by(&self, permutation: &Permutation) -> Self {
        if permutation.natural_and_fixed {
            self.relabel(0, None, false).0
        } else {
            self.relabel(0, Some(permutation), false).0
        }
    }

    pub fn relabel(
        &self,
        first_label: usize,
        permutation: Option<&Permutation>,
        with_mapping: bool,
    ) -> (Self, Option<Vec<usize>>) {
        let n_items = self.n_items();
        if let Some(p) = permutation {
            assert_eq!(n_items, p.n_items());
        };
        let mut labels = Vec::with_capacity(n_items);
        let mut sizes = Vec::with_capacity(first_label + self.active_labels.len() + 1);
        let mut map = HashMap::new();
        let mut next_new_label = first_label;
        for i in 0..n_items {
            let ii = match permutation {
                None => i,
                Some(p) => p.get(i),
            };
            let c = *map.entry(self.allocation[ii]).or_insert_with(|| {
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
            pairs.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
            let x = (0..first_label).chain(pairs.iter().map(|x| x.0)).collect();
            Some(x)
        } else {
            None
        };
        let active_labels = (first_label..sizes.len()).collect();
        (
            Self {
                allocation: labels,
                sizes,
                active_labels,
                available_labels: Vec::new(),
            },
            mapping,
        )
    }

    pub fn relabel_into_slice<T: UnitIncrementor + Copy>(&self, first_label: T, slice: &mut [T]) {
        let mut map = HashMap::new();
        let mut next_new_label = first_label;
        for (old_label, slice_item) in self.allocation.iter().zip(slice.iter_mut()) {
            *slice_item = *map.entry(*old_label).or_insert_with(|| {
                let new_label = next_new_label;
                T::next(&mut next_new_label);
                new_label
            });
        }
    }
}

pub trait FromUsize {
    fn from_usize(x: usize) -> Self;
}

impl FromUsize for i32 {
    #[inline]
    fn from_usize(x: usize) -> Self {
        x as Self
    }
}

impl FromUsize for u16 {
    #[inline]
    fn from_usize(x: usize) -> Self {
        x as Self
    }
}

pub trait UnitIncrementor {
    fn next(x: &mut Self);
}

impl UnitIncrementor for i32 {
    #[inline]
    fn next(x: &mut i32) {
        if *x == i32::MAX {
            panic!("Overflow!");
        }
        *x += 1
    }
}

impl UnitIncrementor for u16 {
    #[inline]
    fn next(x: &mut u16) {
        if *x == u16::MAX {
            panic!("Overflow!");
        }
        *x += 1
    }
}

impl UnitIncrementor for u8 {
    #[inline]
    fn next(x: &mut u8) {
        if *x == u8::MAX {
            panic!("Overflow!");
        }
        *x += 1
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

#[derive(Clone)]
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
            "Clustering { allocation: [0, 0, 0, 0, 0], sizes: [5], active_labels: [0], available_labels: [] }",
        );
        let clustering = Clustering::singleton_clusters(5);
        check_output(
            &clustering,
            "Clustering { allocation: [0, 1, 2, 3, 4], sizes: [1, 1, 1, 1, 1], active_labels: [0, 1, 2, 3, 4], available_labels: [] }",
        );
        let clustering = Clustering::from_vector(vec![2, 2, 4, 3, 4]);
        check_output(
            &clustering,
            "Clustering { allocation: [2, 2, 4, 3, 4], sizes: [0, 0, 2, 1, 2], active_labels: [2, 3, 4], available_labels: [0, 1] }",
        );
        let (clustering, map) = clustering.relabel(1, None, true);
        check_output(
            &clustering,
            "Clustering { allocation: [1, 1, 2, 3, 2], sizes: [0, 2, 2, 1], active_labels: [1, 2, 3], available_labels: [] }",
        );
        check_output(&map, "Some([0, 2, 4, 3])");
    }

    #[test]
    fn test_add_to_unlisted_cluster() {
        let mut clustering = Clustering::singleton_clusters(5);
        let new_label = 6;
        clustering.allocate(1, new_label);
        check_output(&clustering, "Clustering { allocation: [0, 6, 2, 3, 4], sizes: [1, 0, 1, 1, 1, 0, 1], active_labels: [0, 6, 2, 3, 4], available_labels: [5, 1] }");
    }

    #[test]
    fn test_add_to_existing_cluster() {
        let mut clustering = Clustering::singleton_clusters(5);
        let new_label = clustering.new_label();
        clustering.allocate(1, new_label);
        check_output(&clustering, "Clustering { allocation: [0, 5, 2, 3, 4], sizes: [1, 0, 1, 1, 1, 1], active_labels: [0, 5, 2, 3, 4], available_labels: [1] }");
        clustering.allocate(3, 2);
        check_output(&clustering, "Clustering { allocation: [0, 5, 2, 2, 4], sizes: [1, 0, 2, 0, 1, 1], active_labels: [0, 5, 2, 4], available_labels: [1, 3] }");
    }

    #[test]
    fn test_add_to_available_cluster() {
        let mut clustering = Clustering::from_vector(vec![0, 5, 2, 2, 4]);
        clustering.allocate(3, 1);
        check_output(&clustering, "Clustering { allocation: [0, 5, 2, 1, 4], sizes: [1, 1, 1, 0, 1, 1], active_labels: [0, 2, 4, 5, 1], available_labels: [3] }");
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
        clustering.allocate(3, 2);
        clustering.allocate(1, 5);
        check_output(&clustering.active_labels(), "[2, 4, 5]");
    }
}
