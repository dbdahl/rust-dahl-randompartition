// Ewens Pitman attraction partition distribution

use crate::clust::Clustering;
use crate::distr::{
    FullConditional, HasPermutation, NormalizedProbabilityMassFunction, PartitionSampler,
    ProbabilityMassFunction,
};
use crate::perm::Permutation;
use crate::prelude::{Concentration, Discount};

use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use std::slice;

type SimilarityBorrower<'a> = SquareMatrixBorrower<'a>;

#[derive(Debug, Clone)]
pub struct EpaParameters<'a> {
    similarity: SimilarityBorrower<'a>,
    pub permutation: Permutation,
    pub concentration: Concentration,
    pub discount: Discount,
}

impl<'a> EpaParameters<'a> {
    pub fn new(
        similarity: SimilarityBorrower<'a>,
        permutation: Permutation,
        concentration: Concentration,
        discount: Discount,
    ) -> Option<Self> {
        if similarity.n_items() != permutation.n_items() {
            None
        } else {
            Some(Self {
                similarity,
                permutation,
                concentration,
                discount,
            })
        }
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

    /// # Safety
    ///
    /// You're on your own.
    pub unsafe fn from_ptr(data: *mut f64, n_items: usize) -> Self {
        let data = slice::from_raw_parts_mut(data, n_items * n_items);
        Self { data, n_items }
    }

    pub fn n_items(&self) -> usize {
        self.n_items
    }

    /// # Safety
    ///
    /// You're on your own.
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

impl FullConditional for EpaParameters<'_> {
    fn log_full_conditional<'b>(
        &'b self,
        item: usize,
        clustering: &'b Clustering,
    ) -> Vec<(usize, f64)> {
        let mut p = clustering.allocation().clone();
        clustering
            .available_labels_for_reallocation(item)
            .map(|label| {
                p[item] = label;
                (label, engine::<Pcg64Mcg>(self, Some(&p[..]), None).1)
            })
            .collect()
    }
}

impl PartitionSampler for EpaParameters<'_> {
    fn sample<T: Rng>(&self, rng: &mut T) -> Clustering {
        engine(self, None, Some(rng)).0
    }
}

impl ProbabilityMassFunction for EpaParameters<'_> {
    fn log_pmf(&self, partition: &Clustering) -> f64 {
        engine::<Pcg64Mcg>(self, Some(partition.allocation()), None).1
    }
}

impl NormalizedProbabilityMassFunction for EpaParameters<'_> {}

impl HasPermutation for EpaParameters<'_> {
    fn permutation(&self) -> &Permutation {
        &self.permutation
    }
    fn permutation_mut(&mut self) -> &mut Permutation {
        &mut self.permutation
    }
}

pub fn engine<T: Rng>(
    parameters: &EpaParameters,
    target: Option<&[usize]>,
    mut rng: Option<&mut T>,
) -> (Clustering, f64) {
    let ni = parameters.similarity.n_items();
    let concentration = parameters.concentration.get();
    let discount = parameters.discount.get();
    let mut log_probability = 0.0;
    let mut clustering = Clustering::unallocated(ni);
    for i in 0..ni {
        let ii = parameters.permutation.get(i);
        let qt = clustering.n_clusters() as f64;
        let kt = ((i as f64) - discount * qt)
            / parameters
                .similarity
                .sum_of_row_subset(ii, parameters.permutation.as_slice_until(i));
        let labels_and_weights = clustering
            .available_labels_for_allocation_with_target(target, ii)
            .map(|label| {
                let n_items_in_cluster = clustering.size_of(label);
                let weight = if n_items_in_cluster == 0 {
                    concentration + discount * qt
                } else {
                    kt * parameters
                        .similarity
                        .sum_of_row_subset(ii, &clustering.items_of(label)[..])
                };
                (label, weight)
            });
        let (subset_index, log_probability_contribution) = match &mut rng {
            Some(r) => clustering.select(labels_and_weights, false, false, 0, Some(r), true),
            None => clustering.select::<Pcg64Mcg, _>(
                labels_and_weights,
                false,
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
        let mut rng = rand::rng();
        let permutation = Permutation::random(n_items, &mut rng);
        let discount = Discount::new(0.3).unwrap();
        let concentration = Concentration::new_with_discount(1.5, discount).unwrap();
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
            EpaParameters::new(similarity_borrower, permutation, concentration, discount).unwrap();
        let sample_closure = || parameters.sample(&mut rand::rng());
        let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
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
        let mut rng = rand::rng();
        let permutation = Permutation::random(n_items, &mut rng);
        let discount = Discount::new(0.3).unwrap();
        let concentration = Concentration::new_with_discount(1.5, discount).unwrap();
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
            EpaParameters::new(similarity_borrower, permutation, concentration, discount).unwrap();
        let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
        crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
    }
}
