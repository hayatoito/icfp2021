use structopt::StructOpt;

type Result<T> = anyhow::Result<T>;

#[derive(StructOpt, Debug)]
struct Cli {
    // #[structopt(short = "v", parse(from_occurrences))]
    // verbose: u64,
    #[structopt(subcommand)]
    cmd: Cmd,
}

#[derive(StructOpt, Debug)]
enum Cmd {
    #[structopt(name = "hello")]
    Hello,
    #[structopt(name = "problem")]
    Problem { id: u32 },
    #[structopt(name = "download-problem")]
    DownloadProblem { from: u32, inclusive_to: u32 },
    // #[structopt(name = "post-solution")]
    // PostSolution { id: u32 },
    // #[structopt(name = "retrieve-pose-info")]
    // RetrievePoseInfo { problem_id: u32, pose_id: String },
    #[structopt(name = "update-pending")]
    UpdatePending,
    #[structopt(name = "visualize")]
    VisualizeProblem { problem_id: u32 },
    // #[structopt(name = "visualize-solution")]
    // VisualizeSolution { problem_id: u32 },
    #[structopt(name = "solve")]
    Solve { problem_id: u32 },
    // #[structopt(name = "solve-with-manual-input")]
    // SolveWithManualInput {
    //     problem_id: u32,
    //     solution_path: String,
    // },
    #[structopt(name = "solve-and-submit")]
    SolveAndSubmit { problem_id: u32 },
    #[structopt(name = "do-not-solve-and-submit")]
    DoNotSolveAndSubmit { problem_id: u32 },
    #[structopt(name = "solve-all")]
    SolveAll { start: u32, end: u32 },
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Cli::from_args();
    match args.cmd {
        Cmd::Hello => {
            println!("response: {}", icfp2021::api::hello()?);
        }
        Cmd::Problem { id } => {
            println!("response: {}", icfp2021::api::problem(id)?);
        }
        Cmd::DownloadProblem { from, inclusive_to } => {
            icfp2021::api::download_problems(from, inclusive_to)?;
        }
        // Cmd::PostSolution { id } => {
        //     let pose_id = icfp2021::api::post_solution(id)?;
        //     println!("pose_id: {}", pose_id);
        // }
        // Cmd::RetrievePoseInfo {
        //     problem_id,
        //     pose_id,
        // } => {
        //     let pose_info = icfp2021::api::retrieve_pose_info(problem_id, pose_id)?;
        //     println!("pose_info: {}", pose_info);
        // }
        Cmd::UpdatePending => {
            icfp2021::submission::update_pending()?;
        }
        Cmd::VisualizeProblem { problem_id } => {
            icfp2021::plot::visualize_problem(problem_id)?;
        }
        // Cmd::VisualizeSolution { problem_id } => {
        //     icfp2021::plot::visualize_solution(problem_id)?;
        // }
        Cmd::Solve { problem_id } => {
            icfp2021::solver::solve(problem_id)?;
        }
        // Cmd::SolveWithManualInput {
        //     problem_id,
        //     solution_path,
        // } => {
        //     icfp2021::solver::solve_with_manual_input(problem_id, solution_path)?;
        // }
        Cmd::SolveAndSubmit { problem_id } => {
            icfp2021::solver::solve_and_submit(problem_id)?;
        }
        Cmd::DoNotSolveAndSubmit { problem_id } => {
            icfp2021::solver::do_not_solve_and_submit(problem_id)?;
        }
        Cmd::SolveAll { start, end } => {
            icfp2021::solver::solve_all(start..end)?;
        }
    }
    Ok(())
}
