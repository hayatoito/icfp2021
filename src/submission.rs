use crate::prelude::*;

// task/solution/
//   {problem-id}-{pose-id}.json
//   {problem-id}-{pose-id}-judge.json
// task/submit/
//   {problem-id}.json -> ../solution/*.json

#[derive(Serialize, Deserialize, Debug)]
pub struct Registry {
    pub submitted: HashMap<u32, Submission>,
    pub best: HashMap<u32, Submission>,
}

type PoseId = String;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Submission {
    pub solution: Solution,
    pub dislikes: u64,
    pub pose_id: PoseId,
    pub judge: Judge,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Judge {
    pub state: String,
    pub dislikes: Option<u64>,
    pub error: Option<String>,
}

impl Registry {
    pub fn new() -> Result<Registry> {
        Ok(serde_json::from_str(&read_from_task_dir(
            "submission/registry.json",
        )?)?)
    }

    pub fn update_pending(&mut self) -> Result<()> {
        for id in self.pending_submission() {
            info!("update pending: {}", id);
            self.update_if_pending(id)?;
        }
        self.sync()
    }

    fn pending_submission(&self) -> Vec<u32> {
        let mut pendings = vec![];
        for (id, submission) in &self.submitted {
            if submission.judge.state == "PENDING" {
                pendings.push(*id);
            }
        }
        pendings
    }

    fn sync(&self) -> Result<()> {
        info!("synd");
        write_to_task_dir(
            "submission/registry.json",
            &serde_json::to_string_pretty(&self)?,
        )
        .context("write to registry json")?;
        // backup
        write_to_task_dir(
            format!("submission/registry-{}.json", unique_file_name()),
            &serde_json::to_string_pretty(&self)?,
        )
        .context("write to back up registry json")
    }

    fn best_for(&self, problem_id: u32) -> Option<u64> {
        if let Some(submission) = self.best.get(&problem_id) {
            assert_eq!(submission.judge.state, "VALID");
            assert!(submission.judge.dislikes.is_some());
            return submission.judge.dislikes;
        }
        None
    }

    fn update_if_pending(&mut self, problem_id: u32) -> Result<()> {
        if let Some(submission) = self.submitted.get(&problem_id) {
            if submission.judge.state == "PENDING" {
                let pose_info =
                    crate::api::retrieve_pose_info(problem_id, submission.pose_id.clone())?;
                let judge: Judge = serde_json::from_str(&pose_info)?;
                let submission = Submission {
                    judge,
                    ..submission.clone()
                };
                self.submitted.insert(problem_id, submission.clone());
                self.maybe_update_best(problem_id, submission);
                self.sync()?;
            }
        }
        Ok(())
    }

    fn maybe_update_best(&mut self, problem_id: u32, submission: Submission) {
        let dislikes = submission.dislikes;
        let judge = &submission.judge;
        if judge.state == "VALID" {
            assert!(judge.dislikes.is_some());
            let judge_dislikes = judge.dislikes.unwrap();
            if judge_dislikes < self.best_for(problem_id).unwrap_or(u64::MAX) {
                info!("submission is succeeded: Updating best: {:?}", judge);
                self.best.insert(problem_id, submission);
            }
            if dislikes != judge_dislikes {
                warn!(
                    "Different dislike: dislike {}, judge {}",
                    dislikes, judge_dislikes
                );
            }
        } else if judge.state == "PENDING" {
            warn!("PENDING. Consder to retrieve info again later: {:?}", judge);
        } else {
            warn!("INVALID solution: {:?}", judge);
        }
    }

    pub fn submit_if_best(
        &mut self,
        problem_id: u32,
        solution: Solution,
        dislikes: u64,
    ) -> Result<()> {
        self.update_if_pending(problem_id)?;

        let best_dislikes = self.best_for(problem_id).unwrap_or(u64::MAX);
        if dislikes < best_dislikes {
            let pose_id = crate::api::post_solution(problem_id, serde_json::to_string(&solution)?)?;
            let pose_info = crate::api::retrieve_pose_info(problem_id, pose_id.clone())?;
            let judge: Judge = serde_json::from_str(&pose_info)?;
            info!("judge: {:?}", judge);
            let submission = Submission {
                solution,
                dislikes,
                pose_id,
                judge,
            };
            self.submitted.insert(problem_id, submission.clone());
            self.maybe_update_best(problem_id, submission);
            self.sync()?;
        } else {
            info!(
                "solution is not the best: {} vs {}. Skipping submission",
                dislikes, best_dislikes
            );
        }
        Ok(())
    }
}

pub fn update_pending() -> Result<()> {
    let mut registry = Registry::new()?;
    registry.update_pending()
}
