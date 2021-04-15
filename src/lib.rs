#![allow(dead_code)]

extern crate dahl_bellnumber;
extern crate dahl_salso;
extern crate rand;
extern crate statrs;

pub mod clust;
pub mod cpp;
pub mod crp;
pub mod distr;
pub mod epa;
pub mod ffi;
pub mod fixed;
pub mod frp;
pub mod jlp;
pub mod lsp;
pub mod mcmc;
pub mod perm;
pub mod prelude;
pub mod shrink;
pub mod sp;
pub mod testing;
pub mod up;

pub unsafe fn push_into_slice_i32(from_slice: &[usize], to_slice: &mut [i32]) {
    to_slice.iter_mut().zip(from_slice).for_each(|(x, y)| {
        *x = *y as i32;
    });
}
