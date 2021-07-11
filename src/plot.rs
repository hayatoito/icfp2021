use crate::prelude::*;

use super::*;

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

pub trait ToPlot {
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

pub fn visualize_problem(problem_id: u32) -> Result<()> {
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

impl Problem {
    pub fn visualize(&self) -> Result<()> {
        let mut traces = Vec::new();

        // Plot hole.
        let hole_plot = self.hole.to_plot();
        traces.push(hole_plot);

        // Plot figure.
        let edges = self.figure_to_pose(&self.figure.vertices);
        for e in edges {
            traces.push(e.to_plot());
        }

        crate::plot::plot(&traces)
    }
}
