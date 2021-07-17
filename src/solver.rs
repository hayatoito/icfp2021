use crate::prelude::*;

// use crate::geo::*;
use crate::plot::*;
use crate::submission::*;

struct Move {
    v_index: usize,
    current_point: P,
    next_point: P,
}

// polygon.contains requres float.
type Polygon = geo::Polygon<f64>;

pub struct Solver {
    problem_id: u32,
    problem: Problem,
    // For solver.
    width: i64,
    height: i64,
    hole: Vec<P>,
    hole_polygon: Polygon,
    pose: Vec<P>,
    figure_edge_length: Vec<Distance>,
    // rng
    rng: Box<dyn RngCore>,
}

impl Solver {
    fn new(problem_id: u32) -> Result<Solver> {
        let problem = Problem::new(problem_id)?;

        let width = {
            let max_x = problem.hole.iter().map(|p| p.0).max().unwrap();
            let min_x = problem.hole.iter().map(|p| p.0).min().unwrap();
            max_x - min_x
        };
        let height = {
            let max_y = problem.hole.iter().map(|p| p.1).max().unwrap();
            let min_y = problem.hole.iter().map(|p| p.1).min().unwrap();
            max_y - min_y
        };

        let hole: Vec<P> = problem
            .hole
            .iter()
            .map(|(x, y)| (*x, *y).into())
            .collect::<Vec<_>>();
        let hole_polygon = geo::Polygon::new(
            geo::LineString::from(
                hole.iter()
                    .map(|p| p.to_float_coordinate())
                    .collect::<Vec<_>>(),
            ),
            vec![],
        );
        let pose: Vec<P> = problem
            .figure
            .vertices
            .iter()
            .map(|p| (*p).into())
            .collect();
        let figure_edge_length = problem
            .figure
            .edges
            .iter()
            .map(|e| pose[e.0].squared_distance(&pose[e.1]))
            .collect();

        Ok(Solver {
            problem_id,
            problem,
            width,
            height,
            hole,
            hole_polygon,
            pose,
            figure_edge_length,
            rng: Box::new(rand::thread_rng()),
        })
    }

    fn pose_x_y(&self) -> Vec<Point> {
        self.pose.iter().map(|p| p.x_y()).collect()
    }

    fn visualize(&self) -> Result<()> {
        let mut traces = Vec::new();

        // Plot hole.
        let hole_plot = self.problem.hole.to_plot();
        traces.push(hole_plot);

        // Plot pose (solution)
        let edges = self.problem.figure_to_pose(&self.pose_x_y());
        for e in edges {
            traces.push(e.to_plot());
        }

        crate::plot::plot(&traces)
    }

    // Spec 3 (b)
    fn pose_edge_length_ok(&self) -> bool {
        for (i, edge) in self.problem.figure.edges.iter().enumerate() {
            let p0 = self.pose[edge.0];
            let p1 = self.pose[edge.1];

            let d1 = p0.squared_distance(&p1); // new
            let d2 = self.figure_edge_length[i]; // original

            // println!("p0: {:?}, p1: {:?}", p0, p1);
            // println!("d1: {}, d2: {}", d1, d2);

            // | (d1/d2) - 1 | <= e / 1_000_000;
            // 1_000_000 * | d1 - d2| <= d2 * e
            if 1_000_000 * (if d1 > d2 { d1 - d2 } else { d2 - d1 }) > d2 * self.problem.epsilon {
                return false;
            }
        }
        true
    }

    // Spec 3 (c)
    // fn pose_points_in_hole(&self) -> bool {
    //     self.vertices
    //         .iter()
    //         .all(|p| self.hole_polygon.contains(&p.to_float_coordinate()))
    // }

    // Spec 3 (c)
    // fn pose_interect_hole(&self) -> bool {
    //     for edge in &self.problem.figure.edges {
    //         // pose segment
    //         let p0 = self.vertices[edge.0];
    //         let p1 = self.vertices[edge.1];

    //         for j in 0..self.problem.hole.len() {
    //             // hole segment
    //             let h0 = self.problem.hole[j];
    //             let h1 = self.problem.hole[(j + 1) % self.problem.hole.len()];
    //             match (p0, p1).intersect(&(h0, h1)) {
    //                 IntersectResult::Intersect => {
    //                     return true;
    //                 }
    //                 IntersectResult::PointOnSegment => {
    //                     // Note boundary is okay.
    //                 }
    //                 IntersectResult::None => {}
    //             }
    //         }
    //     }
    //     false
    // }

    // Spec 3 (c)
    fn hole_polygon_contains_pose(&self) -> bool {
        self.number_of_invalid_pose_edge_intersect(&self.pose) == 0
    }

    fn check_constraint(&self) -> bool {
        self.pose_edge_length_ok() && self.hole_polygon_contains_pose()
    }

    fn score(&self, pose: &[P]) -> Score {
        10_000 + self.constraint_score(pose) + self.dislikes(pose)
    }

    fn constraint_score(&self, pose: &[P]) -> Score {
        self.number_of_invalid_pose_edge_intersect(pose) as u64 * 1_000_000_000
        //     + (self.invalid_points_score(pose) * 1_000_000_000.0) as u64
            + self.invalid_edge_length_score(pose) * 1
    }

    pub fn dislikes(&self, pose: &[P]) -> Score {
        self.hole
            .iter()
            .map(|p| Self::min_squared_distance(*p, pose))
            .sum()
    }

    fn min_squared_distance(hole_point: P, pose: &[P]) -> Distance {
        assert!(!pose.is_empty());
        pose.iter()
            .map(|p| p.squared_distance(&hole_point))
            .min()
            .unwrap()
    }

    fn number_of_invalid_pose_edge_intersect(&self, pose: &[P]) -> usize {
        self.problem
            .figure
            .edges
            .iter()
            .filter(|edge| {
                // pose segment
                let p0 = pose[edge.0];
                let p1 = pose[edge.1];

                let line = geo::Line::new(p0.to_float_coordinate(), p1.to_float_coordinate());

                // // geo's poliygon contains returns false if boundary
                // !self.hole_polygon.contains(&line)
                !self.hole_polygon.contains_inclusive(&line)
            })
            .count()
    }

    fn invalid_edge_length_score(&self, pose: &[P]) -> u64 {
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
                score += 1_000 + left - right;
            }
        }
        score
    }

    fn next_move(&mut self) -> Move {
        let n = self.pose.len();
        let v_index = self.rng.gen_range(0..n);
        let current_point = self.pose[v_index];
        let delta = P {
            // x: self.rng.gen_range(-100..100),
            // y: self.rng.gen_range(-100..100),
            x: (self.rng.sample::<f64, _>(rand_distr::StandardNormal) * self.width as f64) as i64,
            y: (self.rng.sample::<f64, _>(rand_distr::StandardNormal) * self.height as f64) as i64,
        };
        let next_point = current_point + delta;
        Move {
            v_index,
            current_point,
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

        let mut current_score = self.score(&self.pose);

        while temperature > absolute_temperature {
            // if iteration % 1_000_000 == 0 {
            if iteration % 100_000 == 0 {
                debug!(
                    // "iteration: {}, temperature: {:10.8}, score: {}, e-inter: {}, p-in-hole: {}, e-len: {},  dislike: {}",
                    "ite: {}, t: {:10.8}, score: {}, dislike: {}",
                    iteration,
                    temperature,
                    current_score,
                    // self.number_of_invalid_edge_intersect(&self.vertices) as u64 * 1_000_000,
                    // self.invalid_points_score(&self.vertices) * 1_000_000_000.0,
                    // self.invalid_edge_length_score(&self.vertices) * 100,
                    self.dislikes(&self.pose)
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

            self.pose[v_index] = next_point;

            let next_score = self.score(&self.pose);

            let delta = (next_score as f64) - (current_score as f64);

            if delta < 0.0 || (-delta / temperature).exp() > self.rng.gen::<f64>() {
                current_score = next_score;
            } else {
                // revert to the previous state
                self.pose[v_index] = current_point;
            }
            temperature *= cooling_rate;
            iteration += 1;
        }
        println!("score: {}", self.score(&self.pose));

        self.result()
    }

    fn result(&self) -> Result<SolveResult> {
        if self.check_constraint() {
            Ok(SolveResult::Solved {
                problem_id: self.problem_id,
                solution: self.solution(),
                dislikes: self.dislikes(&self.pose),
            })
        } else {
            Ok(SolveResult::Unresolved)
        }
    }

    fn result_without_check(&self) -> SolveResult {
        SolveResult::Solved {
            problem_id: self.problem_id,
            solution: self.solution(),
            dislikes: self.dislikes(&self.pose),
        }
    }

    fn solution(&self) -> Solution {
        Solution {
            vertices: self.pose_x_y(),
        }
    }

    fn write_solution(&self) -> Result<()> {
        write_to_task_dir(
            &format!("solution/{}-{}.json", self.problem_id, unique_file_name()),
            &serde_json::to_string(&self.solution())?,
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
        solver.pose = solution.vertices.into_iter().map(|p| p.into()).collect();
    }

    let res = solver.solve();
    // let res = solver.result();
    info!("solveresult: {:?}", res);
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

pub fn do_not_solve_and_submit(problem_id: u32) -> Result<()> {
    let mut registory = Registry::new()?;
    let mut solver = Solver::new(problem_id)?;
    let s = read_from_task_dir(&format!("manual/{}.json", problem_id))?;
    let solution: Solution = serde_json::from_str(&s)?;
    solver.pose = solution.vertices.into_iter().map(|p| p.into()).collect();

    if let SolveResult::Solved {
        problem_id,
        solution,
        dislikes,
    } = solver.result_without_check()
    {
        registory.submit_if_best(problem_id, solution, dislikes)?;
    }
    Ok(())
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

// workaroud for polygon contains
trait ContainsInclusive {
    type Target;
    fn contains_inclusive(&self, other: &Self::Target) -> bool;
}

impl ContainsInclusive for Polygon {
    type Target = geo::Line<f64>;

    fn contains_inclusive(&self, line: &Self::Target) -> bool {
        let epsilon = 0.00001;

        let shrink_start = line.start + line.delta() * epsilon;
        let shrink_end = line.end - line.delta() * epsilon;
        // println!("shrink_start: {:?}", shrink_start);
        // println!("shrink_end: {:?}", shrink_end);

        let cross = geo::Coordinate {
            x: line.delta().y,
            y: -line.delta().x,
        };

        let line1 = geo::Line::new(shrink_start + cross * epsilon, shrink_end + cross * epsilon);
        let line2 = geo::Line::new(shrink_start - cross * epsilon, shrink_end - cross * epsilon);

        // println!("line1: {:?}", line1);
        // println!("line2: {:?}", line2);
        self.contains(&line1) || self.contains(&line2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_env_logger() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn read_problem() -> Result<()> {
        let problem = Problem::new(1)?;
        assert_eq!(problem.epsilon, 150000);
        assert_eq!(problem.figure.edges[0], (2, 5));
        assert_eq!(problem.figure.vertices[2], (30, 95));
        assert_eq!(problem.figure.vertices[5], (40, 65));
        Ok(())
    }

    #[test]
    fn read_solution() -> Result<()> {
        let solution = Solution::read_existing_solution(1)?;
        assert_eq!(solution.vertices.len(), 20);

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
        assert!(!solver.hole_polygon_contains_pose());

        assert!(!solver.check_constraint());

        // // solution
        let solution = Solution::read_existing_solution(1)?;
        solver.pose = solution.vertices.into_iter().map(|p| p.into()).collect();
        assert_eq!(solver.dislikes(&solver.pose), 1_097);

        assert!(solver.pose_edge_length_ok());
        assert!(solver.hole_polygon_contains_pose());
        assert!(solver.check_constraint());

        Ok(())
    }

    #[test]
    fn geo_polygon_contains() {
        init_env_logger();

        let polygon = geo::Polygon::new(
            geo::LineString::from(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
            vec![],
        );

        assert!(polygon.contains(&geo::Coordinate { x: 0.5, y: 0.5 }));
        assert!(!polygon.contains(&geo::Coordinate { x: 2.0, y: 2.0 }));
        // false
        assert!(!polygon.contains(&geo::Coordinate { x: 0.0, y: 0.0 }));

        let line = geo::Line::new(
            geo::Coordinate { x: 0.5, y: 0.5 },
            geo::Coordinate { x: 0.8, y: 0.8 },
        );
        assert!(polygon.contains(&line));

        let line = geo::Line::new(
            geo::Coordinate { x: 0.5, y: 0.5 },
            geo::Coordinate { x: 1.0, y: 1.0 },
        );
        assert!(polygon.contains(&line));

        let line = geo::Line::new(
            geo::Coordinate { x: 1.0, y: 0.0 },
            geo::Coordinate { x: 1.0, y: 1.0 },
        );
        // false
        assert!(!polygon.contains(&line));

        // workaround
        assert!(polygon.contains_inclusive(&line));

        // // let relate = polygon.relate(&geo::Coordinate { x: 0.0, y: 0.0 });
        // let relate = polygon.relate(&line);

        // assert!(relate.is_intersects());
        // assert!(!relate.is_disjoint());
        // assert!(!relate.is_contains());
        // assert!(!relate.is_within());
    }
}
