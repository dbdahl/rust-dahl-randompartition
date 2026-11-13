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
use std::sync::Arc;

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

    pub fn view(&mut self) -> SquareMatrixBorrower<'_> {
        SquareMatrixBorrower::from_slice(&mut self.data[..], self.n_items)
    }

    pub fn n_items(&self) -> usize {
        self.n_items
    }
}

#[derive(Debug, Clone)]
pub struct SquareMatrixBorrower<'a> {
    data: Data<'a>,
    n_items: usize,
}

// Shared data makes clones cheap and needs no lifetime.
#[derive(Debug, Clone)]
enum Data<'a> {
    Borrowed(&'a [f64]),
    Shared(Arc<[f64]>),
}

impl std::ops::Index<(usize, usize)> for SquareMatrixBorrower<'_> {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.slice()[self.n_items * j + i]
    }
}

impl SquareMatrixBorrower<'static> {
    pub fn from_shared(data: Arc<[f64]>, n_items: usize) -> Self {
        let n_elements = n_items
            .checked_mul(n_items)
            .expect("matrix dimensions overflow");
        assert_eq!(data.len(), n_elements);
        Self {
            data: Data::Shared(data),
            n_items,
        }
    }
}

impl<'a> SquareMatrixBorrower<'a> {
    pub fn from_slice(data: &'a mut [f64], n_items: usize) -> Self {
        let n_elements = n_items
            .checked_mul(n_items)
            .expect("matrix dimensions overflow");
        assert_eq!(data.len(), n_elements);
        Self {
            data: Data::Borrowed(data),
            n_items,
        }
    }

    /// # Safety
    ///
    /// You're on your own.
    pub unsafe fn from_ptr(data: *mut f64, n_items: usize) -> Self {
        let data = unsafe { slice::from_raw_parts_mut(data, n_items * n_items) };
        Self {
            data: Data::Borrowed(data),
            n_items,
        }
    }

    fn slice(&self) -> &[f64] {
        match &self.data {
            Data::Borrowed(data) => data,
            Data::Shared(data) => data,
        }
    }

    pub fn n_items(&self) -> usize {
        self.n_items
    }

    /// # Safety
    ///
    /// You're on your own.
    pub unsafe fn get_unchecked(&self, (i, j): (usize, usize)) -> &f64 {
        unsafe { self.slice().get_unchecked(self.n_items * j + i) }
    }

    pub fn data(&self) -> &[f64] {
        self.slice()
    }

    pub fn sum_of_triangle(&self) -> f64 {
        let data = self.slice();
        let mut sum = 0.0;
        for i in 0..self.n_items {
            for j in 0..i {
                sum += unsafe { *data.get_unchecked(self.n_items * j + i) };
            }
        }
        sum
    }

    pub fn sum_of_row_subset(&self, row: usize, columns: &[usize]) -> f64 {
        let data = self.slice();
        let mut sum = 0.0;
        for j in columns {
            sum += unsafe { *data.get_unchecked(self.n_items * *j + row) };
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

/// The share of item `row`'s attraction to the `earlier` items that goes to
/// `cluster`, some of them, for when the usual weight isn't finite: the sum
/// of its similarities to `cluster` over the sum to `earlier`.
///
/// - If the similarities to `earlier` are all 0, as when `exp()` underflows,
///   the item is attracted to the earlier items alike, as in the
///   Ewens-Pitman distribution, the limit as they shrink together. The
///   probability of a new cluster, and so the distribution of the number of
///   clusters, still doesn't depend on the similarities.
/// - If their sum overflows, both sums are taken over similarities scaled by
///   a power of 2, which is exact but for terms too small to matter.
fn attraction_share(
    similarity: &SimilarityBorrower,
    row: usize,
    cluster: &[usize],
    earlier: &[usize],
) -> f64 {
    let total = similarity.sum_of_row_subset(row, earlier);
    if total == 0.0 {
        return cluster.len() as f64 / earlier.len() as f64;
    }
    if total.is_infinite() {
        // Each sum is then at most half the largest finite number.
        let scale = 0.5 / earlier.len().next_power_of_two() as f64;
        let sum = |items: &[usize]| {
            items
                .iter()
                .map(|&j| similarity[(row, j)] * scale)
                .sum::<f64>()
        };
        return sum(cluster) / sum(earlier);
    }
    similarity.sum_of_row_subset(row, cluster) / total
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
        let earlier = parameters.permutation.as_slice_until(i);
        let numerator = (i as f64) - discount * qt;
        let total = parameters.similarity.sum_of_row_subset(ii, earlier);
        let kt = numerator / total;
        let labels_and_weights = clustering
            .available_labels_for_allocation_with_target(target, ii)
            .map(|label| {
                let n_items_in_cluster = clustering.size_of(label);
                let items = &clustering.items_of(label)[..];
                // The first item always starts a cluster.
                let weight = if i == 0 {
                    1.0
                } else if n_items_in_cluster == 0 {
                    concentration + discount * qt
                } else if kt.is_finite() && total.is_finite() {
                    kt * parameters.similarity.sum_of_row_subset(ii, items)
                } else {
                    numerator * attraction_share(&parameters.similarity, ii, items, earlier)
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
    #[should_panic(expected = "matrix dimensions overflow")]
    fn test_from_shared_rejects_dimension_overflow() {
        let n_items = 1usize << (usize::BITS / 2);
        SquareMatrixBorrower::from_shared(Arc::from([]), n_items);
    }

    #[test]
    #[should_panic(expected = "matrix dimensions overflow")]
    fn test_from_slice_rejects_dimension_overflow() {
        let n_items = 1usize << (usize::BITS / 2);
        SquareMatrixBorrower::from_slice(&mut [], n_items);
    }

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

    // The degenerate sums that `attraction_share` handles: whole numbers,
    // so that scaling them by a power of 2 is exact, even to subnormal
    // numbers, and whose sums of two or more overflow when scaled by 2^1020.
    #[test]
    fn test_degenerate_similarities() {
        let n_items = 4;
        let mut rng = rand::rng();
        let permutation = Permutation::random(n_items, &mut rng);
        let discount = Discount::new(0.3).unwrap();
        let concentration = Concentration::new_with_discount(1.5, discount).unwrap();
        let (a, b, c, d, e, f) = (15.0, 10.0, 8.0, 9.0, 12.0, 14.0);
        let base = [
            0.0, a, b, c, //
            a, 0.0, d, e, //
            b, d, 0.0, f, //
            c, e, f, 0.0,
        ];
        let log_pmfs = |data: &[f64]| -> Vec<f64> {
            let similarity = SquareMatrixBorrower::from_shared(data.into(), n_items);
            let parameters =
                EpaParameters::new(similarity, permutation.clone(), concentration, discount)
                    .unwrap();
            let mut rng = rand::rng();
            for _ in 0..100 {
                parameters.sample(&mut rng);
            }
            Clustering::iter(n_items)
                .map(|p| parameters.log_pmf(&Clustering::from_vector(p)))
                .collect()
        };
        let assert_close = |x: &[f64], y: &[f64]| {
            for (x, y) in x.iter().zip(y) {
                assert!((x - y).abs() < 1e-12, "{x} differs from {y}");
            }
        };
        let expected = log_pmfs(&base);
        // The smallest subnormal number, and a scale at which sums overflow.
        for factor in [f64::from_bits(1), 2f64.powi(1020)] {
            assert_close(&log_pmfs(&base.map(|x| x * factor)), &expected);
        }
        // With every similarity 0, as with equal ones, the distribution is
        // the Ewens-Pitman distribution.
        assert_close(&log_pmfs(&[0.0; 16]), &log_pmfs(&[1.0; 16]));
        // An item with no similarity to the others.
        let mut isolated = base;
        for j in 1..n_items {
            isolated[j] = 0.0;
            isolated[n_items * j] = 0.0;
        }
        let total: f64 = log_pmfs(&isolated).iter().map(|x| x.exp()).sum();
        assert!((total - 1.0).abs() < 1e-12, "{total}");
    }

    #[test]
    fn test_pmf() {
        let n_items = 4;
        let mut rng = rand::rng();
        let permutation = Permutation::random(n_items, &mut rng);
        let discount = Discount::new(0.3).unwrap();
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
        let shared = SquareMatrixBorrower::from_shared(similarity.data().into(), n_items);
        let borrowed = similarity.view();
        for similarity_borrower in [borrowed, shared] {
            // The concentration is greater than minus the discount.
            for concentration in [1.5, -0.2] {
                let concentration =
                    Concentration::new_with_discount(concentration, discount).unwrap();
                let parameters = EpaParameters::new(
                    similarity_borrower.clone(),
                    permutation.clone(),
                    concentration,
                    discount,
                )
                .unwrap();
                let log_prob_closure = |clustering: &mut Clustering| parameters.log_pmf(clustering);
                crate::testing::assert_pmf_sums_to_one(n_items, log_prob_closure, 0.0000001);
            }
        }
    }
}
