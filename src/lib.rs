#![allow(dead_code)]

extern crate dahl_bellnumber;
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
pub mod distr;
pub mod ffi;
pub mod fixed;
pub mod perm;
pub mod prelude;
pub mod sp;
pub mod testing;
pub mod up;
pub mod wgt;

pub unsafe fn push_into_slice_i32(from_slice: &[usize], to_slice: &mut [i32]) {
    to_slice.iter_mut().zip(from_slice).for_each(|(x, y)| {
        *x = *y as i32;
    });
}
