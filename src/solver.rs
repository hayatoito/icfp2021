use crate::prelude::*;

use chrono::prelude::*;
use std::io::Write;

mod plot {
    use super::*;

    fn unique_file_name() -> String {
        let local: DateTime<Local> = Local::now();
        local.format("%Y-%m-%d-%H%M%S-%f").to_string()
    }

    pub fn plot<T>(data: &T) -> Result<()>
    where
        T: Serialize,
    {
        let mut path = std::env::temp_dir();
        path.push(&format!("plot-{}.html", unique_file_name()));
        write_html(&path, data)?;
        webbrowser::open(path.to_str().unwrap())?;
        // Ok(path)
        Ok(())
    }

    fn write_html<T: ?Sized>(path: impl AsRef<Path>, data: &T) -> Result<()>
    where
        T: Serialize,
    {
        let html = format!(
            r###"
<!DOCTYPE html>
<head>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<!-- <div id="myDiv" style="width:1280px;height:720px"></div> -->
<div id="myDiv" style="width:800px;height:600px"></div>
<script>
const myDiv = document.getElementById('myDiv');
Plotly.newPlot(myDiv, {});
</script>
    "###,
            serde_json::to_string(&data).unwrap(),
        );
        let mut file = std::fs::File::create(path.as_ref())?;
        file.write_all(html.as_bytes())?;
        Ok(())
    }
}

type Cord = i64;
type Point = (Cord, Cord);

#[derive(Serialize, Deserialize, Debug)]
struct Problem {
    hole: Vec<Point>,
    figure: Figure,
    epsilon: u64,
}

trait ToPlot {
    fn to_plot(&self) -> serde_json::Value;
}

impl ToPlot for Vec<Point> {
    fn to_plot(&self) -> serde_json::Value {
        let mut x = self.iter().map(|p| p.0).collect::<Vec<_>>();
        let mut y = self.iter().map(|p| p.1).collect::<Vec<_>>();
        x.push(self[0].0);
        y.push(self[0].1);
        serde_json::json!({
            "x": x,
            "y": y,
            "type": "scatter",
        })
    }
}

impl ToPlot for (Point, Point) {
    fn to_plot(&self) -> serde_json::Value {
        serde_json::json!({
            "x": [self.0.0, self.1.0],
            "y": [self.0.1, self.1.1],
            "type": "scatter",
        })
    }
}

type Score = u64;
type Distance = u64;

type Pose = Vec<Point>;

trait SquaredDistance {
    fn squared_distance(&self, other: &Self) -> Distance;
}

impl SquaredDistance for Point {
    fn squared_distance(&self, other: &Self) -> Distance {
        // TODO: might not fit in u64.
        ((self.0 - other.0).pow(2) + (self.1 - other.1).pow(2)) as u64
    }
}

impl Problem {
    pub fn new(id: u32) -> Result<Problem> {
        let json = read_from_task_dir(&format!("problem/{}.json", id))?;
        Ok(serde_json::from_str(&json)?)
    }

    fn figure_to_pose(&self, vertices: &[Point]) -> Vec<(Point, Point)> {
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

    fn visualize(&self) -> Result<()> {
        let mut traces = Vec::new();

        // Plot hole.
        let hole_plot = self.hole.to_plot();
        traces.push(hole_plot);

        // Plot figure.
        let edges = self.figure_to_pose(&self.figure.vertices);
        for e in edges {
            traces.push(e.to_plot());
        }

        plot::plot(&traces)
    }

    fn min_squared_distance(point_in_hole: Point, pose: &Pose) -> Distance {
        assert!(!pose.is_empty());
        pose.iter()
            .map(|p| p.squared_distance(&point_in_hole))
            .min()
            .unwrap()
    }

    fn dislike(&self, pose: &Pose) -> Score {
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

// index in vertices.
type IndexInVerticles = usize;
type Edge = (IndexInVerticles, IndexInVerticles);

#[derive(Serialize, Deserialize, Debug)]
struct Figure {
    edges: Vec<Edge>,
    vertices: Vec<Point>,
}

impl Figure {
    fn edge_length(&self) -> Vec<Distance> {
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

/// Solution
// [[file:~/share/rust/icfp2021/task/solution/1.json::{]]
#[derive(Serialize, Deserialize, Debug)]
struct Solution {
    vertices: Vec<Point>,
}

impl Solution {
    fn read_existing_solution(id: u32) -> Result<Solution> {
        let json = read_from_task_dir(&format!("solution/{}.json", id))?;
        Ok(serde_json::from_str(&json)?)
    }
}

pub fn visualize(problem_id: u32) -> Result<()> {
    let problem = Problem::new(problem_id)?;
    problem.visualize()
}

pub fn visualize_solution(problem_id: u32) -> Result<()> {
    let problem = Problem::new(problem_id)?;
    let solution = Solution::read_existing_solution(1)?;

    let mut traces = Vec::new();

    // Plot hole.
    let hole_plot = problem.hole.to_plot();
    traces.push(hole_plot);

    // Plot figure.
    let edges = problem.figure_to_pose(&problem.figure.vertices);
    for e in edges {
        traces.push(e.to_plot());
    }

    // Plot pose (solution)
    let edges = problem.figure_to_pose(&solution.vertices);
    for e in edges {
        traces.push(e.to_plot());
    }

    plot::plot(&traces)
}

struct Solver {
    problem: Problem,
    vertices: Vec<Point>,
    figure_edge_length: Vec<Distance>,
}

impl Solver {
    fn new(problem_id: u32) -> Result<Solver> {
        let problem = Problem::new(problem_id)?;
        let vertices = problem.figure.vertices.clone();
        let figure_edge_length = problem.figure.edge_length();

        Ok(Solver {
            problem,
            vertices,
            figure_edge_length,
        })
    }

    // Spec 3 (b)
    fn pose_edge_length_ok(&self) -> bool {
        for (i, edge) in self.problem.figure.edges.iter().enumerate() {
            let p0 = self.vertices[edge.0];
            let p1 = self.vertices[edge.1];

            let d1 = p0.squared_distance(&p1); // new
            let d2 = self.figure_edge_length[i]; // original

            // | (d1/d2) - 1 | <= e / 1_000_000;
            // 1_000_000 * | d1 - d2| <= d2 * e
            if 1_000_000 * (if d1 > d2 { d1 - d2 } else { d2 - d1 }) > d2 * self.problem.epsilon {
                return false;
            }
        }
        true
    }

    // Spec 3 (c)
    fn pose_points_in_hole(&self) -> bool {
        self.vertices
            .iter()
            .all(|p| is_inside(&self.problem.hole, *p))
    }

    // Spec 3 (c)
    fn pose_interect_hole(&self) -> bool {
        for edge in &self.problem.figure.edges {
            // pose segment
            let p0 = self.vertices[edge.0];
            let p1 = self.vertices[edge.1];

            for j in 0..self.problem.hole.len() {
                // hole segment
                let h0 = self.problem.hole[j];
                let h1 = self.problem.hole[(j + 1) % self.problem.hole.len()];
                match (p0, p1).intersect(&(h0, h1)) {
                    IntersectResult::Intersect => {
                        return true;
                    }
                    IntersectResult::PointOnSegment => {
                        // Note boundary is okay.
                    }
                    IntersectResult::None => {}
                }
            }
        }
        false
    }

    fn check_constraint(&self) -> bool {
        self.pose_edge_length_ok() && self.pose_points_in_hole() && !self.pose_interect_hole()
    }
}

// Geometry

// https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
type Segment = (Point, Point);

enum IntersectResult {
    PointOnSegment,
    Intersect,
    None,
}

trait Intersect {
    fn intersect(&self, other: &Self) -> IntersectResult;
}

// Given three colinear points p, q, r, the function checks if
// point q lies on line segment 'pr'
fn on_segment(p: Point, q: Point, r: Point) -> bool {
    q.0 <= p.0.max(r.0) && q.0 >= p.0.min(r.0) && q.1 <= p.1.max(r.1) && q.1 >= p.1.min(r.1)
}

fn orientation(p: Point, q: Point, r: Point) -> u32 {
    // See https://www.geeksforgeeks.org/orientation-3-ordered-points/
    // for details of below formula.
    let val = (q.1 - p.1) * (r.0 - q.0) - (q.0 - p.0) * (r.1 - q.1);

    if val == 0 {
        return 0; // colinear
    }

    // clock or counterclock wise
    if val > 0 {
        1
    } else {
        2
    }
}

// / The main function that returns true if line segment 'p1q1'
// // and 'p2q2' intersect.
// bool doIntersect(Point p1, Point q1, Point p2, Point q2)
impl Intersect for Segment {
    fn intersect(&self, other: &Segment) -> IntersectResult {
        let p1 = self.0;
        let q1 = self.1;

        let p2 = other.0;
        let q2 = other.1;
        // Find the four orientations needed for general and
        // special cases
        let o1 = orientation(p1, q1, p2);
        let o2 = orientation(p1, q1, q2);
        let o3 = orientation(p2, q2, p1);
        let o4 = orientation(p2, q2, q1);

        // println!("{} != {}, {} != {}", o1, o2, o3, o4);

        // General case
        if o1 != o2 && o3 != o4 {
            if o1 == 0 && on_segment(p1, p2, q1) {
                return IntersectResult::PointOnSegment;
            }
            if o2 == 0 && on_segment(p1, q2, q1) {
                return IntersectResult::PointOnSegment;
            }
            if o3 == 0 && on_segment(p2, p1, q2) {
                return IntersectResult::PointOnSegment;
            }
            if o4 == 0 && on_segment(p2, q1, q2) {
                return IntersectResult::PointOnSegment;
            }

            return IntersectResult::Intersect;
        }

        // Special Cases
        // p1, q1 and p2 are colinear and p2 lies on segment p1q1
        if o1 == 0 && on_segment(p1, p2, q1) {
            return IntersectResult::PointOnSegment;
        }

        // p1, q1 and q2 are colinear and q2 lies on segment p1q1
        if o2 == 0 && on_segment(p1, q2, q1) {
            return IntersectResult::PointOnSegment;
        }

        // p2, q2 and p1 are colinear and p1 lies on segment p2q2
        if o3 == 0 && on_segment(p2, p1, q2) {
            return IntersectResult::PointOnSegment;
        }

        // p2, q2 and q1 are colinear and q1 lies on segment p2q2
        if o4 == 0 && on_segment(p2, q1, q2) {
            return IntersectResult::PointOnSegment;
        }
        return IntersectResult::None; // Doesn't fall in any of the above cases
    }
}

// https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
// Returns true if the point p lies inside the polygon[] with n vertices
fn is_inside(polygon: &[Point], p: Point) -> bool {
    // There must be at least 3 vertices in polygon[]
    assert!(polygon.len() >= 3);

    // Create a point for line segment from p to infinite
    let extreme = ((i32::MAX / 10) as i64, p.1);

    // Count intersections of the above line with sides of polygon
    let mut count = 0;
    for i in 0..polygon.len() {
        let next = (i + 1) % polygon.len();

        // Check if the line segment from 'p' to 'extreme' intersects
        // with the line segment from 'polygon[i]' to 'polygon[next]'
        // if (doIntersect(polygon[i], polygon[next], p, extreme))
        match (polygon[i], polygon[next]).intersect(&(p, extreme)) {
            // IntersectResult::Intersect => {
            //     count += 1;
            // }
            IntersectResult::Intersect | IntersectResult::PointOnSegment => {
                count += 1;
                // If the point 'p' is colinear with line segment 'i-next',
                // then check if it lies on segment. If it lies, return true,
                // otherwise false
                if orientation(polygon[i], p, polygon[next]) == 0 {
                    // warn!("p is on_segmetn?: {:?}", p);
                    println!(
                        "p is on_segment: {:?} on {:?}",
                        p,
                        (polygon[i], polygon[next])
                    );
                    return on_segment(polygon[i], p, polygon[next]);
                }
            }
            IntersectResult::None => {}
        }
    }

    // Return true if count is odd, false otherwise
    if count % 2 == 0 {
        println!("p is outside of polygon: {:?}, count: {}", p, count);
    }
    count % 2 == 1
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn read_problem() -> Result<()> {
        let problem = Problem::new(1)?;
        assert_eq!(problem.epsilon, 150000);
        assert_eq!(problem.figure.edges[0], (2, 5));
        assert_eq!(problem.figure.vertices[2], (30, 95));
        assert_eq!(problem.figure.vertices[5], (40, 65));
        assert_eq!((30, 95).squared_distance(&(40, 65)), 1000);
        assert_eq!(
            problem.figure.edge_length(),
            vec![
                1000, 900, 425, 100, 650, 125, 125, 100, 125, 125, 650, 100, 425, 900, 1000, 100,
                725, 725, 100, 100, 125, 100, 125, 125, 100, 125, 125, 125, 125, 125
            ]
        );
        Ok(())
    }

    #[test]
    fn read_solution() -> Result<()> {
        let solution = Solution::read_existing_solution(1)?;
        assert_eq!(solution.vertices.len(), 20);

        let problem = Problem::new(1)?;
        assert_eq!(problem.dislike(&solution.vertices), 3_704);
        Ok(())
    }

    #[test]
    fn squared_distance() {
        let p1 = (0, 0);
        let p2 = (3, 4);
        assert_eq!(p1.squared_distance(&p2), 25);
    }

    #[test]
    fn solver_constraint() -> Result<()> {
        let mut solver = Solver::new(1)?;
        assert!(solver.pose_edge_length_ok());
        assert!(!solver.pose_points_in_hole());
        assert!(solver.pose_interect_hole());

        assert!(!solver.check_constraint());

        // // solution
        let solution = Solution::read_existing_solution(1)?;
        solver.vertices = solution.vertices;

        assert!(solver.pose_edge_length_ok());
        assert!(solver.pose_points_in_hole());
        assert!(!solver.pose_interect_hole());

        // assert!(solver.check_constraint());

        Ok(())
    }

    #[test]
    fn prelude_test_dummy() {
        // assert_eq!(0, 1 - 1);
    }
}
