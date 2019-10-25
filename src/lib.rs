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

enum TargetOrRandom<'a> {
    Target(&'a Partition),
    Random(ThreadRng),
}
