use std::collections::HashMap;
use std::ops::Index;

#[derive(Debug, Clone)]
pub struct Clustering {
    pub labels: Vec<usize>,
    pub sizes: Vec<usize>,
    expired_labels: Vec<usize>,
    last_label: Option<usize>,
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
    pub fn one_cluster(n_items: usize) -> Self {
        Self {
            labels: vec![0; n_items],
            sizes: {
                let mut sizes = Vec::with_capacity(K);
                sizes.push(n_items);
                sizes
            },
            expired_labels: Vec::new(),
            last_label: None,
        }
    }

    pub fn n_clusters(n_items: usize) -> Self {
        Self {
            labels: (0..n_items).collect(),
            sizes: vec![1; n_items],
            expired_labels: Vec::new(),
            last_label: None,
        }
    }

    pub fn from_vector(labels: Vec<usize>) -> Self {
        let mut x = Self {
            labels,
            sizes: Vec::with_capacity(K),
            expired_labels: Vec::new(),
            last_label: None,
        };
        for label in &x.labels {
            if *label >= x.sizes.len() {
                x.sizes.resize(*label + 1, 0)
            }
            x.sizes[*label] += 1;
        }
        for (index, size) in x.sizes.iter().enumerate() {
            if *size == 0 {
                x.expired_labels.push(index)
            }
        }
        x
    }

    pub fn from_slice(labels: &[usize]) -> Self {
        Self::from_vector(labels.to_vec())
    }

    pub fn get(&mut self, item: usize) -> usize {
        self.labels[item]
    }

    pub fn set(&mut self, item: usize, label: usize) {
        let old_label = self[item];
        if old_label == label {
            return;
        }
        match self.sizes.get_mut(label) {
            Some(new_size) => {
                if *new_size == 0 {
                    match self.last_label {
                        Some(last_label) => {
                            if last_label != label {
                                panic!("Attempting to allocate to an invalid empty cluster.  Call 'new_cluster' first.");
                            }
                        }
                        None => {
                            panic!("Attempting to allocate to an invalid empty cluster.  Call 'new_cluster' first.");
                        }
                    }
                } else {
                    if let Some(last_label) = self.last_label {
                        self.expired_labels.push(last_label)
                    }
                }
                self.last_label = None;
                self.labels[item] = label;
                *new_size += 1;
            }
            None => panic!("Attempting to allocate to a non-existing cluster."),
        }
        let old_label_size = &mut self.sizes[old_label];
        *old_label_size -= 1;
        if *old_label_size == 0 {
            self.expired_labels.push(old_label)
        }
    }

    pub fn new_cluster(&mut self) -> usize {
        match self.last_label {
            Some(last_label) => last_label,
            None => {
                let new_label = match self.expired_labels.pop() {
                    Some(available_label) => available_label,
                    None => {
                        self.sizes.push(0);
                        self.sizes.len() - 1
                    }
                };
                self.last_label = Some(new_label);
                new_label
            }
        }
    }

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

    pub fn standardize(&self, with_mapping: bool) -> (Self, Option<Vec<usize>>) {
        let n_items = self.labels.len();
        let mut labels = Vec::with_capacity(n_items);
        let mut sizes = Vec::with_capacity(K);
        let mut map = HashMap::new();
        let mut next_new_label = 0;
        for j in 0..n_items {
            let c = *map.entry(self.labels[j]).or_insert_with(|| {
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
        (
            Self {
                labels,
                sizes,
                expired_labels: Vec::new(),
                last_label: None,
            },
            mapping,
        )
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
            "Clustering { labels: [0, 0, 0, 0, 0], sizes: [5], expired_labels: [], last_label: None }",
        );
        let clustering = Clustering::n_clusters(5);
        check_output(
            &clustering,
            "Clustering { labels: [0, 1, 2, 3, 4], sizes: [1, 1, 1, 1, 1], expired_labels: [], last_label: None }",
        );
        let clustering = Clustering::from_slice(&[2, 2, 4, 3, 4]);
        check_output(
            &clustering,
            "Clustering { labels: [2, 2, 4, 3, 4], sizes: [0, 0, 2, 1, 2], expired_labels: [0, 1], last_label: None }",
        );
        let (clustering, map) = clustering.standardize(true);
        check_output(
            &clustering,
            "Clustering { labels: [0, 0, 1, 2, 1], sizes: [2, 2, 1], expired_labels: [], last_label: None }",
        );
        check_output(&map, "Some([2, 4, 3])");
    }

    #[test]
    #[should_panic]
    fn test_add_to_nonexisting_cluster() {
        let mut clustering = Clustering::n_clusters(5);
        let new_label = 5;
        clustering.set(1, new_label);
        println!("{:?}", clustering);
    }

    #[test]
    fn test_new_cluster_twice() {
        let mut clustering = Clustering::n_clusters(5);
        check_output(&clustering, "Clustering { labels: [0, 1, 2, 3, 4], sizes: [1, 1, 1, 1, 1], expired_labels: [], last_label: None }");
        clustering.new_cluster();
        check_output(&clustering, "Clustering { labels: [0, 1, 2, 3, 4], sizes: [1, 1, 1, 1, 1, 0], expired_labels: [], last_label: Some(5) }");
        clustering.new_cluster();
        check_output(&clustering, "Clustering { labels: [0, 1, 2, 3, 4], sizes: [1, 1, 1, 1, 1, 0], expired_labels: [], last_label: Some(5) }");
    }

    #[test]
    fn test_add_to_existing_cluster() {
        let mut clustering = Clustering::n_clusters(5);
        let new_label = clustering.new_cluster();
        clustering.set(1, new_label);
        check_output(&clustering, "Clustering { labels: [0, 5, 2, 3, 4], sizes: [1, 0, 1, 1, 1, 1], expired_labels: [1], last_label: None }");
        clustering.new_cluster();
        check_output(&clustering, "Clustering { labels: [0, 5, 2, 3, 4], sizes: [1, 0, 1, 1, 1, 1], expired_labels: [], last_label: Some(1) }");
        clustering.set(3, 2);
    }

    #[test]
    #[should_panic]
    fn test_add_to_improper_cluster() {
        let mut clustering = Clustering::from_slice(&[0, 5, 2, 2, 4]);
        clustering.new_cluster();
        clustering.set(3, 1);
    }

    #[test]
    fn test_items_of() {
        let clustering = Clustering::from_slice(&[2, 2, 4, 3, 4]);
        check_output(&clustering.items_of(4), "[2, 4]");
    }
}
