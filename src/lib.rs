#![allow(dead_code)]

extern crate dahl_partition;
extern crate dahl_salso;
extern crate rand;
extern crate statrs;

mod clust;
//pub mod cpp;
pub mod crp;
//pub mod epa;
//pub mod frp;
//pub mod lsp;
pub mod mcmc;
//pub mod nggp;
pub mod prelude;
pub mod testing;

use crate::clust::Clustering;
use rand::prelude::*;

pub enum TargetOrRandom<'a, T: Rng> {
    Target(&'a mut dahl_partition::Partition),
    Random(&'a mut T),
}

pub enum TargetOrRandom2<'a, T: Rng> {
    Target(Clustering),
    Random(&'a mut T),
}
