// pub use lazy_static::*;
pub use anyhow::{bail, ensure, Context, Result};
pub use log::*;
pub use serde::{Deserialize, Serialize};
pub use std::collections::{HashMap, HashSet, VecDeque};
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

#[cfg(test)]
mod tests {

    #[test]
    fn prelude_test_dummy() {
        // assert_eq!(0, 1 - 1);
    }
}
