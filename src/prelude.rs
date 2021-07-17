// pub use lazy_static::*;
pub use anyhow::{bail, ensure, Context, Result};
pub use chrono::prelude::*;
pub use geo::algorithm::contains::Contains;
pub use log::*;
pub use rand::Rng;
pub use rand::RngCore;
pub use rayon::prelude::*;
pub use serde::{Deserialize, Serialize};
pub use std::collections::{HashMap, HashSet, VecDeque};
pub use std::io::Write;
pub use std::ops::Deref;
pub use std::ops::Range;
pub use std::path::{Path, PathBuf};
pub use std::rc::Rc;

pub fn read_from_task_dir(task_relative_path: impl AsRef<Path>) -> Result<String> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("task");
    path.push(task_relative_path);
    Ok(std::fs::read_to_string(path)?)
}

pub fn write_to_task_dir(task_relative_path: impl AsRef<Path>, content: &str) -> Result<()> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("task");
    path.push(task_relative_path);
    Ok(std::fs::write(path, content)?)
}

pub fn unique_file_name() -> String {
    let local: DateTime<Local> = Local::now();
    local.format("%Y-%m-%d-%H%M%S-%f").to_string()
}

// for icfp 2021
pub type Score = u64;
pub type Distance = u64;

pub type Coord = i64;

pub type Point = (Coord, Coord);

pub type P = geo::Coordinate<Coord>;

pub trait ToFloatCoordinate {
    fn to_float_coordinate(&self) -> geo::Coordinate<f64>;
}

impl ToFloatCoordinate for P {
    fn to_float_coordinate(&self) -> geo::Coordinate<f64> {
        geo::Coordinate {
            x: self.x as f64,
            y: self.y as f64,
        }
    }
}

// Use distance_2

pub trait SquaredDistance {
    fn squared_distance(&self, other: &Self) -> Distance;
}

impl SquaredDistance for Point {
    fn squared_distance(&self, other: &Self) -> Distance {
        // TODO: might not fit in u64.
        ((self.0 - other.0).pow(2) + (self.1 - other.1).pow(2)) as u64
    }
}

impl SquaredDistance for P {
    fn squared_distance(&self, other: &Self) -> Distance {
        ((self.x - other.x).pow(2) + (self.y - other.y).pow(2)) as u64
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Problem {
    pub hole: Vec<Point>,
    pub figure: Figure,
    pub epsilon: u64,
}

// index in vertices.
type IndexInVerticles = usize;
type Edge = (IndexInVerticles, IndexInVerticles);

#[derive(Serialize, Deserialize, Debug)]
pub struct Figure {
    pub edges: Vec<Edge>,
    pub vertices: Vec<Point>,
}

impl Problem {
    pub fn new(id: u32) -> Result<Problem> {
        let json = read_from_task_dir(&format!("problem/{}.json", id))?;
        Ok(serde_json::from_str(&json)?)
    }

    pub fn figure_to_pose(&self, vertices: &[Point]) -> Vec<(Point, Point)> {
        let mut edges = Vec::new();
        for i in 0..self.figure.edges.len() {
            let start_index = self.figure.edges[i].0;
            let end_index = self.figure.edges[i].1;
            let start_point = vertices[start_index];
            let end_point = vertices[end_index];
            edges.push((start_point, end_point));
        }
        edges
    }
}

/// Solution
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Solution {
    pub vertices: Vec<Point>,
}

impl Solution {
    pub fn read_existing_solution(id: u32) -> Result<Solution> {
        let json = read_from_task_dir(&format!("solution/{}.json", id))?;
        Ok(serde_json::from_str(&json)?)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn prelude_test_dummy() {
        // assert_eq!(0, 1 - 1);
    }
}
