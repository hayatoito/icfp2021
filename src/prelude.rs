// pub use lazy_static::*;
pub use anyhow::{bail, ensure, Context, Result};
pub use log::*;
pub use serde::{Deserialize, Serialize};
pub use std::collections::{HashMap, HashSet, VecDeque};
pub use std::path::{Path, PathBuf};
pub use std::rc::Rc;

type Cord = i64;
type Point = (Cord, Cord);

#[derive(Serialize, Deserialize, Debug)]
struct Problem {
    hole: Vec<Point>,
    figures: Figures,
    epsilon: u32,
}

// index in vertices.
type IndexInVerticles = u32;
type Edge = (IndexInVerticles, IndexInVerticles);

#[derive(Serialize, Deserialize, Debug)]
struct Figures {
    edges: Vec<Edge>,
    vertices: Vec<Point>,
}

#[cfg(test)]
mod tests {

    #[test]
    fn prelude_test_dummy() {
        // assert_eq!(0, 1 - 1);
    }
}
