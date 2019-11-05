#![allow(dead_code)]

extern crate dahl_partition;
extern crate dahl_salso;
extern crate rand;
extern crate statrs;

pub mod crp;
pub mod frp;
pub mod mcmc;
pub mod nggp;
pub mod prelude;

use dahl_partition::*;
use rand::prelude::*;

pub enum TargetOrRandom<'a, T: Rng> {
    Target(&'a mut Partition),
    Random(&'a mut T),
}

impl<'a, T: Rng> TargetOrRandom<'a, T> {
    pub fn get_rng(&mut self) -> &mut T {
        match self {
            TargetOrRandom::Random(rng) => rng,
            _ => panic!("Not available."),
        }
    }
}
