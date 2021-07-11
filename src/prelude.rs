// pub use lazy_static::*;
pub use anyhow::{bail, ensure, Context, Result};
pub use chrono::prelude::*;
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

// 2021
pub type Score = u64;
pub type Distance = u64;

pub type Cord = i64;
pub type Point = (Cord, Cord);

#[derive(Debug, PartialEq, PartialOrd, Clone, Copy, Default)]
pub struct P(pub Point);

impl P {
    pub fn x(&self) -> i64 {
        self.0 .0
    }
    pub fn y(&self) -> i64 {
        self.0 .1
    }

    pub fn dot(&self, rhs: &P) -> i64 {
        self.x() * rhs.x() + self.y() * rhs.y()
    }
}

impl Deref for P {
    type Target = Point;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::Add for P {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        P((self.x() + rhs.x(), self.y() + rhs.y()))
    }
}

impl std::ops::Mul<Cord> for P {
    type Output = Self;

    fn mul(self, rhs: Cord) -> Self::Output {
        P((self.x() * rhs, self.y() * rhs))
    }
}

impl std::ops::Div<Cord> for P {
    type Output = Self;

    fn div(self, rhs: Cord) -> Self::Output {
        P((self.x() / rhs, self.y() / rhs))
    }
}

impl std::ops::Sub for P {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        P((self.x() - rhs.x(), self.y() - rhs.y()))
    }
}

pub type Segment = (Point, Point);

pub trait SquaredDistance {
    fn squared_distance(&self, other: &Self) -> Distance;
}

impl SquaredDistance for Point {
    fn squared_distance(&self, other: &Self) -> Distance {
        // TODO: might not fit in u64.
        ((self.0 - other.0).pow(2) + (self.1 - other.1).pow(2)) as u64
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

impl Figure {
    pub fn edge_length(&self) -> Vec<Distance> {
        self.edges
            .iter()
            .map(|e| {
                let p0 = self.vertices[e.0];
                let p1 = self.vertices[e.1];
                p0.squared_distance(&p1)
            })
            .collect()
    }
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

    fn min_squared_distance(point_in_hole: Point, pose: &Pose) -> Distance {
        assert!(!pose.is_empty());
        pose.iter()
            .map(|p| p.squared_distance(&point_in_hole))
            .min()
            .unwrap()
    }

    pub fn dislike(&self, pose: &Pose) -> Score {
        self.hole
            .iter()
            .map(|p| Self::min_squared_distance(*p, pose))
            .sum()
    }

    /*
    fn pose_example(&self) -> Pose {
        vec![
            (21, 28),
            (31, 28),
            (31, 87),
            (29, 41),
            (44, 43),
            (58, 70),
            (38, 79),
            (32, 31),
            (36, 50),
            (39, 40),
            (66, 77),
            (42, 29),
            (46, 49),
            (49, 38),
            (39, 57),
            (69, 66),
            (41, 70),
            (39, 60),
            (42, 25),
            (40, 35),
        ]
    }
    */
}

pub type Pose = Vec<Point>;

/// Solution
// [[file:~/share/rust/icfp2021/task/solution/1.json::{]]
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
