use crate::prelude::*;
use reqwest::StatusCode;

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

pub fn hello() -> Result<String> {
    get("/api/hello")
}
