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
    #[structopt(name = "sub1")]
    Sub2 { a: String },
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Cli::from_args();
    match args.cmd {
        Cmd::Hello => {
            let response = icfp2021::api::hello()?;
            println!("response {}", response);
        }
        Cmd::Sub2 { a } => {
            println!("cmd1: {}", a);
        }
    }
    Ok(())
}
