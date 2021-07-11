use crate::prelude::*;

use crate::geo::*;
use crate::plot::*;
use crate::submission::*;

struct Move {
    v_index: usize,
    current_point: Point,
    next_point: Point,
}

pub struct Solver {
    problem_id: u32,
    problem: Problem,
    vertices: Vec<Point>,
    figure_edge_length: Vec<Distance>,
    rng: Box<dyn RngCore>,
}

impl Solver {
    fn new(problem_id: u32) -> Result<Solver> {
        let problem = Problem::new(problem_id)?;
        let vertices = problem.figure.vertices.clone();
        let figure_edge_length = problem.figure.edge_length();

        Ok(Solver {
            problem_id,
            problem,
            vertices,
            figure_edge_length,
            rng: Box::new(rand::thread_rng()),
        })
    }

    fn visualize(&self) -> Result<()> {
        let mut traces = Vec::new();

        // Plot hole.
        let hole_plot = self.problem.hole.to_plot();
        traces.push(hole_plot);

        // Plot pose (solution)
        let edges = self.problem.figure_to_pose(&self.vertices);
        for e in edges {
            traces.push(e.to_plot());
        }

        crate::plot::plot(&traces)
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

    fn score(&self, pose: &Pose) -> Score {
        self.constrait_score(pose) + self.problem.dislike(pose)
    }

    fn constrait_score(&self, pose: &Pose) -> Score {
        self.number_of_invalid_edge_intersect(pose) as u64 * 1_000_000
            + (self.invalid_points_score(pose) * 1_000_000_000.0) as u64
            + self.invalid_edge_length_score(pose) * 100
    }

    fn invalid_points_score(&self, pose: &[Point]) -> f64 {
        let mut score = 0.0;
        for p in pose {
            if is_inside(&self.problem.hole, *p) {
                continue;
            }

            let mut min_dist = f64::MAX;
            // mini distance of point p to a hole segment.
            let hole_len = self.problem.hole.len();
            for i in 0..hole_len {
                let next = (i + 1) % hole_len;
                let hole_segment = (self.problem.hole[i], self.problem.hole[next]);
                min_dist = min_dist.min(p.distance_to_segment(&&hole_segment));
            }
            score += min_dist;
        }
        score
    }

    fn number_of_invalid_edge_intersect(&self, pose: &[Point]) -> usize {
        let mut count = 0;
        for edge in &self.problem.figure.edges {
            // pose segment
            let p0 = pose[edge.0];
            let p1 = pose[edge.1];

            for j in 0..self.problem.hole.len() {
                // hole segment
                let h0 = self.problem.hole[j];
                let h1 = self.problem.hole[(j + 1) % self.problem.hole.len()];

                match (p0, p1).intersect(&(h0, h1)) {
                    // IntersectResult::Intersect => {
                    //     // count multiple times for every intersections....
                    //     count += 1;
                    //     // TODO: break here?
                    // }
                    // IntersectResult::PointOnSegment => {
                    //     // check middle of edge is inside of polygon
                    //     // TODO: pick more random points on (p0, p1)
                    //     let middle = (p0.0 + p1.1 / 2, (p0.1 + p1.1) / 2);
                    //     if !is_inside(&self.problem.hole, middle) {
                    //         count += 1;
                    //     }
                    // }
                    IntersectResult::Intersect | IntersectResult::PointOnSegment => {
                        let p01 = P(p1) - P(p0);
                        for i in 0..10 {
                            let middle = P(p0) + p01 * i / 10;
                            if !is_inside(&self.problem.hole, middle.0) {
                                count += 1;
                            }
                        }
                    }
                    IntersectResult::None => {}
                }
            }
        }
        count
    }

    fn invalid_edge_length_score(&self, pose: &[Point]) -> u64 {
        let mut score = 0;
        for (i, edge) in self.problem.figure.edges.iter().enumerate() {
            let p0 = pose[edge.0];
            let p1 = pose[edge.1];

            let d1 = p0.squared_distance(&p1); // new
            let d2 = self.figure_edge_length[i]; // original

            // | (d1/d2) - 1 | <= e / 1_000_000;
            // 1_000_000 * | d1 - d2| <= d2 * e

            let left = 1_000_000 * (if d1 > d2 { d1 - d2 } else { d2 - d1 });
            let right = d2 * self.problem.epsilon;
            if left > right {
                score += left - right;
            }
        }
        score
    }

    fn dislike(&self) -> Score {
        self.problem.dislike(&self.vertices)
    }

    fn next_move(&mut self) -> Move {
        let n = self.vertices.len();
        let v_index = self.rng.gen_range(0..n);
        let p = self.vertices[v_index];
        let next_point = (
            // p.0 + self.rng.gen_range(-5..5),
            // p.1 + self.rng.gen_range(-5..5),
            p.0 + self.rng.gen_range(-10..10),
            p.1 + self.rng.gen_range(-10..10),
            // p.0 + self.rng.gen_range(-100..100),
            // p.1 + self.rng.gen_range(-100..100),
        );
        Move {
            v_index,
            current_point: p,
            next_point,
        }
    }

    fn solve(&mut self) -> Result<SolveResult> {
        // let mut route = (0..n).collect::<Route>();
        let mut iteration = 0;
        // let mut temperature = 100000.0;
        // let cooling_rate = 0.9999999;
        // let absolute_temperature = 0.000001;

        let mut temperature = 1_000_000.0;
        let cooling_rate = 0.99999;
        let absolute_temperature = 0.00001;

        let mut current_score = self.score(&self.vertices);

        while temperature > absolute_temperature {
            // if iteration % 1_000_000 == 0 {
            if iteration % 100_000 == 0 {
                debug!(
                    "iteration: {}, temperature: {:10.8}, score: {}, e-inter: {}, p-in-hole: {}, e-len: {},  dislike: {}",
                    iteration,
                    temperature,
                    current_score,
                    self.number_of_invalid_edge_intersect(&self.vertices) as u64 * 1_000_000,
                    self.invalid_points_score(&self.vertices) * 1_000_000_000.0,
                    self.invalid_edge_length_score(&self.vertices) * 100,
                    self.dislike()
                );
                if log::log_enabled!(log::Level::Debug) {
                    self.visualize().unwrap();
                }
            }
            let Move {
                v_index,
                current_point,
                next_point,
            } = self.next_move();

            self.vertices[v_index] = next_point;

            let next_score = self.score(&self.vertices);

            let delta = (next_score as f64) - (current_score as f64);

            if delta < 0.0 || (-delta / temperature).exp() > self.rng.gen::<f64>() {
                current_score = next_score;
            } else {
                // revert to the previous state
                self.vertices[v_index] = current_point;
            }
            temperature *= cooling_rate;
            iteration += 1;
        }
        println!("score: {}", self.score(&self.vertices));

        if self.check_constraint() {
            Ok(SolveResult::Solved {
                problem_id: self.problem_id,
                solution: Solution {
                    vertices: self.vertices.clone(),
                },
                dislikes: self.dislike(),
            })
        } else {
            Ok(SolveResult::Unresolved)
        }
    }

    fn write_solution(&self) -> Result<()> {
        let solution = Solution {
            vertices: self.vertices.clone(),
        };
        write_to_task_dir(
            &format!("solution/{}-{}.json", self.problem_id, unique_file_name()),
            &serde_json::to_string(&solution)?,
        )
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SolveResult {
    Solved {
        problem_id: u32,
        solution: Solution,
        dislikes: u64,
    },
    Unresolved,
}

pub fn solve(problem_id: u32) -> Result<SolveResult> {
    let mut solver = Solver::new(problem_id)?;
    if let Ok(s) = read_from_task_dir(&format!("manual/{}.json", problem_id)) {
        warn!("Using manual input for {}", problem_id);
        let solution: Solution = serde_json::from_str(&s)?;
        solver.vertices = solution.vertices;
    }

    let res = solver.solve();
    info!("solveresult: {:?}", res);
    solver.visualize()?;
    solver.write_solution()?;
    res
}

pub fn solve_with_manual_input(
    problem_id: u32,
    solution_path: impl AsRef<Path>,
) -> Result<SolveResult> {
    let solution: Solution = serde_json::from_str(&std::fs::read_to_string(solution_path)?)?;
    let mut solver = Solver::new(problem_id)?;
    solver.vertices = solution.vertices;

    let res = solver.solve();
    solver.visualize()?;
    solver.write_solution()?;
    res
}

pub fn solve_and_submit(problem_id: u32) -> Result<bool> {
    let mut registory = Registry::new()?;
    match solve(problem_id)? {
        SolveResult::Solved {
            problem_id,
            solution,
            dislikes,
        } => {
            registory.submit_if_best(problem_id, solution, dislikes)?;
            Ok(true)
        }
        SolveResult::Unresolved => {
            eprintln!("can not solve: problem_id: {}", problem_id);
            Ok(false)
        }
    }
}

fn start_submission_registry(
    max_message: u32,
) -> (
    std::thread::JoinHandle<()>,
    std::sync::mpsc::Sender<SolveResult>,
) {
    use std::sync::mpsc;
    use std::thread;

    let (tx, rx) = mpsc::channel();

    let handle = thread::spawn(move || {
        info!("registry thead is starting...");
        let mut registory = Registry::new().unwrap();
        let mut solved = vec![];
        for _ in 0..max_message {
            match rx.recv().unwrap() {
                SolveResult::Solved {
                    problem_id,
                    solution,
                    dislikes,
                } => {
                    solved.push(problem_id);
                    if let Err(e) = registory.submit_if_best(problem_id, solution, dislikes) {
                        warn!("submission failed?: {:?}", e);
                    }
                }
                SolveResult::Unresolved => {}
            }
        }
        println!("solved: {:?}", solved);
    });
    (handle, tx)
}

pub fn solve_all(range: Range<u32>) -> Result<()> {
    let (handle, tx) = start_submission_registry(range.end - range.start);
    let problems = range.map(|id| (id, tx.clone())).collect::<Vec<(u32, _)>>();
    problems.into_par_iter().for_each(|(id, tx)| {
        if let Ok(solve_result) = solve(id) {
            tx.send(solve_result).unwrap();
        }
    });
    handle.join().unwrap();
    Ok(())
}

#[cfg(test)]
mod tests {

    fn init_env_logger() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

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
        init_env_logger();

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
