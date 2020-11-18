#![allow(dead_code)]

extern crate dahl_salso;
extern crate rand;
extern crate statrs;

pub mod clust;
pub mod cpp;
pub mod crp;
pub mod epa;
pub mod frp;
pub mod lsp;
pub mod mcmc;
//pub mod nggp;
pub mod prelude;
pub mod prior;
pub mod testing;

pub unsafe fn push_into_slice_i32(from_slice: &[usize], to_slice: &mut [i32]) {
    to_slice.iter_mut().zip(from_slice).for_each(|(x, y)| {
        *x = *y as i32;
    });
}
