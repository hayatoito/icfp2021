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
    #[structopt(name = "post-solution")]
    PostSolution { id: u32 },
    #[structopt(name = "retrieve-pose-info")]
    RetrievePoseInfo { problem_id: u32, pose_id: String },
    #[structopt(name = "visualize")]
    Visualize { problem_id: u32 },
    #[structopt(name = "visualize-solution")]
    VisualizeSolution { problem_id: u32 },
    #[structopt(name = "solve")]
    Solve { problem_id: u32 },
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
        Cmd::PostSolution { id } => {
            let pose_id = icfp2021::api::post_solution(id)?;
            println!("pose_id: {}", pose_id);
        }
        Cmd::RetrievePoseInfo {
            problem_id,
            pose_id,
        } => {
            let pose_info = icfp2021::api::retrieve_pose_info(problem_id, pose_id)?;
            println!("pose_info: {}", pose_info);
        }
        Cmd::Visualize { problem_id } => {
            icfp2021::solver::visualize(problem_id)?;
        }
        Cmd::VisualizeSolution { problem_id } => {
            icfp2021::solver::visualize_solution(problem_id)?;
        }
        Cmd::Solve { problem_id } => {
            icfp2021::solver::solve(problem_id)?;
        }
    }
    Ok(())
}
