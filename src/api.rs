use crate::prelude::*;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

fn api_token() -> Result<String> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("task/api-token");
    Ok(std::fs::read_to_string(path)?.trim().to_string())
}

fn get(path: &str) -> Result<String> {
    let server_url = "https://poses.live/";
    let auth_header = format!("Bearer {}", api_token()?);

    let url = format!("{}{}", server_url, path);
    info!("get url: {}", url);

    let client = reqwest::blocking::Client::new();
    let response = client
        .get(&url)
        .header(reqwest::header::AUTHORIZATION, auth_header)
        .send()?;
    match response.status() {
        StatusCode::OK => Ok(response.text()?),
        _ => {
            error!("Unexpected server response:");
            error!("HTTP code: {}", response.status());
            bail!("response error: {} {}", response.status(), response.text()?)
        }
    }
}

fn post(path: &str, body: String) -> Result<String> {
    let server_url = "https://poses.live/";
    let auth_header = format!("Bearer {}", api_token()?);

    let url = format!("{}{}", server_url, path);
    info!("get url: {}", url);

    let client = reqwest::blocking::Client::new();
    let response = client
        .post(&url)
        .header(reqwest::header::AUTHORIZATION, auth_header)
        .body(body)
        .send()?;
    match response.status() {
        StatusCode::OK => Ok(response.text()?),
        _ => {
            error!("Unexpected server response:");
            error!("HTTP code: {}", response.status());
            bail!("response error: {} {}", response.status(), response.text()?)
        }
    }
}

pub fn hello() -> Result<String> {
    get("/api/hello")
}

pub fn problem(id: u32) -> Result<String> {
    get(&format!("/api/problems/{}", id))
}

// type Poseid = i64;

fn read_solution(id: u32) -> Result<String> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push(&format!("task/solution/{}.json", id));
    Ok(std::fs::read_to_string(path)?)
}

#[derive(Serialize, Deserialize, Debug)]
struct PoseIdResponse {
    id: String,
}

pub fn post_solution(problem_id: u32) -> Result<String> {
    let pose_id_response = post(
        &format!("/api/problems/{}/solutions", problem_id),
        read_solution(problem_id)?,
    )?;
    info!("pose_id_response: {}", pose_id_response);

    // retrieve pose info
    let PoseIdResponse { id: pose_id } = serde_json::from_str(&pose_id_response)?;

    retrieve_pose_info(problem_id, pose_id)
}

pub fn retrieve_pose_info(problem_id: u32, pose_id: String) -> Result<String> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push(&format!("task/pose-info/{}-{}.json", problem_id, pose_id));

    let pose_info = get(&format!(
        "/api/problems/{}/solutions/{}",
        problem_id, pose_id
    ))?;
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push(&format!(
        "task/pose-info/problem-{}_pose-{}.json",
        problem_id, pose_id
    ));
    std::fs::write(path, &pose_info)?;
    Ok(pose_info)
}

pub fn download_problems(from: u32, inclusive_to: u32) -> Result<()> {
    for id in from..=inclusive_to {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push(&format!("task/problem/{}.json", id));
        let response = get(&format!("/api/problems/{}", id))?;
        std::fs::write(path, response)?;
    }
    Ok(())
}
