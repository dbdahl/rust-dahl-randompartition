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

// Initial maximum number of clusters
const K: usize = 20;

impl Clustering {
    pub fn new(n_items: usize) -> Self {
        Self {
            labels: vec![usize::max_value(); n_items],
            sizes: {
                let mut sizes = Vec::with_capacity(K);
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
                let mut sizes = Vec::with_capacity(K);
                sizes.push(n_items);
                sizes.push(0);
                sizes
            },
            active_labels: {
                let mut active_labels = Vec::with_capacity(K);
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
                let mut sizes = Vec::with_capacity(K);
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
            sizes: Vec::with_capacity(K),
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

    pub fn from_slice(labels: &[usize]) -> Self {
        Self::from_vector(labels.to_vec())
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
        let size = self.sizes[label];
        size
    }

    pub fn new_label(&self) -> usize {
        self.sizes.len() - 1
    }

    pub fn active_labels(&self) -> &Vec<usize> {
        &self.active_labels
    }

    pub fn available_labels_iter(&self) -> ClusterLabelsIterator {
        ClusterLabelsIterator {
            iter: self.active_labels.iter(),
            new_label: self.new_label(),
            done: false,
        }
    }

    pub fn select_randomly<T: Rng, S: std::iter::Iterator<Item = f64>>(
        &self,
        probs: S,
        rng: &mut T,
    ) -> usize {
        let dist = WeightedIndex::new(probs).unwrap();
        let index = dist.sample(rng);
        if index == self.n_clusters() {
            self.new_label()
        } else {
            self.active_labels[index]
        }
    }

    pub fn get(&mut self, item: usize) -> usize {
        self.labels[item]
    }

    pub fn reassign(&mut self, item: usize, label: usize) {
        let old_label = self.labels[item];
        if old_label == label {
            return;
        }
        self.assign(item, label);
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

    pub fn assign(&mut self, item: usize, label: usize) {
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
        with_mapping: bool,
        permutation: Option<&Permutation>,
    ) -> (Self, Option<Vec<usize>>) {
        let n_items = self.n_items();
        if let Some(p) = permutation {
            assert_eq!(n_items, p.len());
        };
        let mut labels = Vec::with_capacity(n_items);
        let mut sizes = Vec::with_capacity(K);
        let mut map = HashMap::new();
        let mut next_new_label = 0;
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
            if c == sizes.len() {
                sizes.resize(c + 1, 1)
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
        let active_labels = (0..sizes.len()).collect();
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

pub struct ClusterLabelsIterator<'a> {
    iter: std::slice::Iter<'a, usize>,
    new_label: usize,
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
                    Some(self.new_label)
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
        let clustering = Clustering::from_slice(&[2, 2, 4, 3, 4]);
        check_output(
            &clustering,
            "Clustering { labels: [2, 2, 4, 3, 4], sizes: [0, 0, 2, 1, 2, 0], active_labels: [2, 3, 4], expired_labels: [] }",
        );
        let (clustering, map) = clustering.standardize(true, None);
        check_output(
            &clustering,
            "Clustering { labels: [0, 0, 1, 2, 1], sizes: [2, 2, 1, 0], active_labels: [0, 1, 2], expired_labels: [] }",
        );
        check_output(&map, "Some([2, 4, 3])");
    }

    #[test]
    #[should_panic]
    fn test_add_to_nonexisting_cluster() {
        let mut clustering = Clustering::singleton_clusters(5);
        let new_label = 6;
        clustering.reassign(1, new_label);
        println!("{:?}", clustering);
    }

    #[test]
    fn test_add_to_existing_cluster() {
        let mut clustering = Clustering::singleton_clusters(5);
        let new_label = clustering.new_label();
        clustering.reassign(1, new_label);
        check_output(&clustering, "Clustering { labels: [0, 5, 2, 3, 4], sizes: [1, 0, 1, 1, 1, 1, 0], active_labels: [0, 2, 3, 4, 5], expired_labels: [1] }");
        clustering.new_label();
        check_output(&clustering, "Clustering { labels: [0, 5, 2, 3, 4], sizes: [1, 0, 1, 1, 1, 1, 0], active_labels: [0, 2, 3, 4, 5], expired_labels: [1] }");
        clustering.reassign(3, 2);
    }

    #[test]
    #[should_panic]
    fn test_add_to_improper_cluster() {
        let mut clustering = Clustering::from_slice(&[0, 5, 2, 2, 4]);
        clustering.new_label();
        clustering.reassign(3, 1);
    }

    #[test]
    fn test_items_of() {
        let clustering = Clustering::from_slice(&[2, 2, 4, 3, 4]);
        check_output(&clustering.items_of(4), "[2, 4]");
    }

    #[test]
    fn test_unique_labels() {
        let mut clustering = Clustering::from_slice(&[2, 2, 4, 3, 4]);
        check_output(&clustering.active_labels(), "[2, 3, 4]");
        clustering.reassign(3, 2);
        clustering.reassign(1, 5);
        check_output(&clustering.active_labels(), "[2, 4, 5]");
    }
}
