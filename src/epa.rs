// Ewens Pitman attraction partition distribution

use crate::clust::Clustering;
use crate::distr::{PartitionSampler, PredictiveProbabilityFunctionOld};
use crate::perm::Permutation;
use crate::prelude::*;
use crate::prior::PartitionLogProbability;

use rand::prelude::*;
use rand_isaac::IsaacRng;
use std::slice;

type SimilarityBorrower<'a> = SquareMatrixBorrower<'a>;

#[derive(Debug, Clone)]
pub struct EpaParameters<'a> {
    similarity: SimilarityBorrower<'a>,
    permutation: Permutation,
    mass: Mass,
    discount: Discount,
}

impl<'a> EpaParameters<'a> {
    pub fn new(
        similarity: SimilarityBorrower<'a>,
        permutation: Permutation,
        mass: Mass,
        discount: Discount,
    ) -> Option<Self> {
        if similarity.n_items() != permutation.n_items() {
            None
        } else {
            Some(Self {
                similarity,
                permutation,
                mass,
                discount,
            })
        }
    }

    pub fn shuffle_permutation<T: Rng>(&mut self, rng: &mut T) {
        self.permutation.shuffle(rng);
    }
}

/// A data structure representing a square matrix.
///
#[derive(Debug)]
pub struct SquareMatrix {
    data: Vec<f64>,
    n_items: usize,
}

impl SquareMatrix {
    pub fn zeros(n_items: usize) -> Self {
        Self {
            data: vec![0.0; n_items * n_items],
            n_items,
        }
    }

    pub fn ones(n_items: usize) -> Self {
        Self {
            data: vec![1.0; n_items * n_items],
            n_items,
        }
    }

    pub fn identity(n_items: usize) -> Self {
        let ni1 = n_items + 1;
        let n2 = n_items * n_items;
        let mut data = vec![0.0; n2];
        let mut i = 0;
        while i < n2 {
            data[i] = 1.0;
            i += ni1
        }
        Self { data, n_items }
    }

    pub fn data(&self) -> &[f64] {
        &self.data[..]
    }

    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data[..]
    }

    pub fn view(&mut self) -> SquareMatrixBorrower {
        SquareMatrixBorrower::from_slice(&mut self.data[..], self.n_items)
    }

    pub fn n_items(&self) -> usize {
        self.n_items
    }
}

#[derive(Debug, Clone)]
pub struct SquareMatrixBorrower<'a> {
    data: &'a [f64],
    n_items: usize,
}

impl std::ops::Index<(usize, usize)> for SquareMatrixBorrower<'_> {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.data[self.n_items * j + i]
    }
}

impl<'a> SquareMatrixBorrower<'a> {
    pub fn from_slice(data: &'a mut [f64], n_items: usize) -> Self {
        assert_eq!(data.len(), n_items * n_items);
        Self { data, n_items }
    }

    pub unsafe fn from_ptr(data: *mut f64, n_items: usize) -> Self {
        let data = slice::from_raw_parts_mut(data, n_items * n_items);
        Self { data, n_items }
    }

    pub fn n_items(&self) -> usize {
        self.n_items
    }

    pub unsafe fn get_unchecked(&self, (i, j): (usize, usize)) -> &f64 {
        self.data.get_unchecked(self.n_items * j + i)
    }

    pub fn data(&self) -> &[f64] {
        self.data
    }

    pub fn sum_of_triangle(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.n_items {
            for j in 0..i {
                sum += unsafe { *self.get_unchecked((i, j)) };
            }
        }
        sum
    }

    pub fn sum_of_row_subset(&self, row: usize, columns: &[usize]) -> f64 {
        let mut sum = 0.0;
        for j in columns {
            sum += unsafe { *self.get_unchecked((row, *j)) };
        }
        sum
    }
}

impl<'a> PredictiveProbabilityFunctionOld for EpaParameters<'a> {
    fn log_predictive_probability(
        &self,
        item_index: usize,
        subset_index: usize,
        clustering: &Clustering,
    ) -> f64 {
        let mut p = clustering.allocation().clone();
        p[item_index] = subset_index;
        engine::<IsaacRng>(self, Some(&p[..]), None).1
    }
}

impl<'a> PartitionSampler for EpaParameters<'a> {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering {
        engine(self, None, Some(rng)).0
    }
}

impl<'a> PartitionLogProbability for EpaParameters<'a> {
    fn log_probability(&self, partition: &Clustering) -> f64 {
        engine::<IsaacRng>(self, Some(partition.allocation()), None).1
    }
    fn is_normalized(&self) -> bool {
        true
    }
}

pub fn engine<T: Rng>(
    parameters: &EpaParameters,
    target: Option<&[usize]>,
    mut rng: Option<&mut T>,
) -> (Clustering, f64) {
    let ni = parameters.similarity.n_items();
    let mass = parameters.mass.unwrap();
    let discount = parameters.discount.unwrap();
    let mut log_probability = 0.0;
    let mut clustering = Clustering::unallocated(ni);
    for i in 0..ni {
        let ii = parameters.permutation.get(i);
        let qt = clustering.n_clusters() as f64;
        let kt = ((i as f64) - discount * qt)
            / parameters
                .similarity
                .sum_of_row_subset(ii, parameters.permutation.slice_until(i));
        let labels_and_weights = clustering
            .available_labels_for_allocation_with_target(target, ii)
            .map(|label| {
                let n_items_in_cluster = clustering.size_of(label);
                let weight = if n_items_in_cluster == 0 {
                    mass + discount * qt
                } else {
                    kt * parameters
                        .similarity
                        .sum_of_row_subset(ii, &clustering.items_of(label)[..])
                };
                (label, weight)
            });
        let (subset_index, log_probability_contribution) = match &mut rng {
            Some(r) => clustering.select(labels_and_weights, false, 0, Some(r), true),
            None => clustering.select::<IsaacRng, _>(
                labels_and_weights,
                false,
                target.unwrap()[ii],
                None,
                true,
            ),
        };
        log_probability += log_probability_contribution;
        clustering.allocate(ii, subset_index);
    }
    (clustering, log_probability)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goodness_of_fit_constructive() {
        let n_items = 4;
        let mut rng = thread_rng();
        let permutation = Permutation::random(n_items, &mut rng);
        let discount = 0.3;
        let mass = Mass::new_with_variable_constraint(1.5, discount);
        let discount = Discount::new(discount);
        let mut similarity = SquareMatrix::zeros(n_items);
        {
            let data = similarity.data_mut();
            data[0] = 0.0;
            data[4] = 0.9;
            data[8] = 0.6;
            data[12] = 0.3;
            data[1] = 0.9;
            data[5] = 0.0;
            data[9] = 0.1;
            data[13] = 0.2;
            data[2] = 0.6;
            data[6] = 0.1;
            data[10] = 0.0;
            data[14] = 0.6;
            data[3] = 0.3;
            data[7] = 0.2;
            data[11] = 0.6;
            data[15] = 0.0;
        }
        let similarity_borrower = similarity.view();
        let parameters =
            EpaParameters::new(similarity_borrower, permutation, mass, discount).unwrap();
        let sample_closure = || parameters.sample(&mut thread_rng());
        let log_prob_closure = |clustering: &mut Clustering| parameters.log_probability(clustering);
        crate::testing::assert_goodness_of_fit(
            10000,
            n_items,
            sample_closure,
            log_prob_closure,
            1,
            0.001,
        );
    }

    #[test]
    fn test_pmf() {
        let n_items = 4;
        let mut rng = thread_rng();
        let permutation = Permutation::random(n_items, &mut rng);
        let discount = 0.3;
        let mass = Mass::new_with_variable_constraint(1.5, discount);
        let discount = Discount::new(discount);
        let mut similarity = SquareMatrix::zeros(n_items);
        {
            let data = similarity.data_mut();
            data[0] = 0.0;
            data[4] = 0.9;
            data[8] = 0.6;
            data[12] = 0.3;
            data[1] = 0.9;
            data[5] = 0.0;
            data[9] = 0.1;
            data[13] = 0.2;
            data[2] = 0.6;
            data[6] = 0.1;
            data[10] = 0.0;
            data[14] = 0.6;
            data[3] = 0.3;
            data[7] = 0.2;
            data[11] = 0.6;
            data[15] = 0.0;
        }
        let similarity_borrower = similarity.view();
        let parameters =
            EpaParameters::new(similarity_borrower, permutation, mass, discount).unwrap();
        let log_prob_closure = |clustering: &mut Clustering| parameters.log_probability(clustering);
        crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__epaparameters_new(
    n_items: i32,
    similarity_ptr: *mut f64,
    permutation_ptr: *const i32,
    use_natural_permutation: i32,
    mass: f64,
    discount: f64,
) -> *mut EpaParameters<'static> {
    let ni = n_items as usize;
    let similarity = SquareMatrixBorrower::from_ptr(similarity_ptr, ni);
    let permutation = if use_natural_permutation != 0 {
        Permutation::natural_and_fixed(ni)
    } else {
        let permutation_slice = slice::from_raw_parts(permutation_ptr, ni);
        let permutation_vector: Vec<usize> =
            permutation_slice.iter().map(|x| *x as usize).collect();
        Permutation::from_vector(permutation_vector).unwrap()
    };
    let d = Discount::new(discount);
    let m = Mass::new_with_variable_constraint(mass, discount);
    // First we create a new object.
    let obj = EpaParameters::new(similarity, permutation, m, d).unwrap();
    // Then copy it to the heap (so we have a stable pointer to it).
    let boxed_obj = Box::new(obj);
    // Then return a pointer by converting our `Box<_>` into a raw pointer
    Box::into_raw(boxed_obj)
}

#[no_mangle]
pub unsafe extern "C" fn dahl_randompartition__epaparameters_free(obj: *mut EpaParameters) {
    // As a rule of thumb, freeing a null pointer is just a noop.
    if obj.is_null() {
        return;
    }
    // Convert the raw pointer back to a Box<_>
    let boxed = Box::from_raw(obj);
    // Then explicitly drop it (optional)
    drop(boxed);
}
