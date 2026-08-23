//! 产物路径与作业状态查询。
//!

use anyhow::{bail, Result};
use serde::Serialize;
use std::path::PathBuf;
use std::time::Duration;
use ts_rs::TS;

use crate::sh;

/// models/ 的位置。
///
/// 默认**从可执行文件的位置反推**，不写死任何绝对路径 —— 整棵树以 `models/`
/// 为根放进版本库后，clone 到哪里都能跑：
///
/// ```text
/// <models>/code/web-demo/server/target/release/wave-demo
///                                        ^ exe 所在目录，往上数 5 层就是 <models>
/// ```
///
/// `WAVE_ROOT` 可以覆盖（`start.sh` 就是显式传的）。返回的一定是**绝对路径** ——
/// shell 脚本和 `fs::read` 用的是同一个字符串，留着 `~` 的话前者能展开、后者不能。
pub fn repo_root() -> String {
    if let Ok(p) = std::env::var("WAVE_ROOT") {
        return p;
    }
    if let Some(r) = root_from_exe() {
        return r;
    }
    // 兜底：连自己在哪都问不出来时，按当前目录算（从 web-demo/ 起跑的话是对的）
    std::env::current_dir()
        .map(|d| d.join("../..").to_string_lossy().into_owned())
        .unwrap_or_else(|_| ".".into())
}

/// `<models>/code/web-demo/server/target/release/wave-demo` → `<models>`
fn root_from_exe() -> Option<String> {
    let exe = std::env::current_exe().ok()?.canonicalize().ok()?;
    let mut d = exe.parent()?; // release
    for _ in 0..5 {
        d = d.parent()?; // target → server → web-demo → code → models
    }
    Some(d.to_string_lossy().into_owned())
}

/// checkpoint 里 enabled 的四个通道。`Uy`（准二维，信号≈噪声）和 `nut`
/// （FUNWAVE 无粘，结构上没有先验）在这条线上是关掉的，所以不给选。
pub const FIELDS: &[&str] = &["alpha", "Ux", "Uz", "p_rgh"];

const STYLE: &str = "tri";
const STATUS_TIMEOUT: Duration = Duration::from_secs(30);

// ───────────────────────────── 对外结构 ─────────────────────────────

#[derive(Debug, Clone, Serialize, TS, Default)]
#[ts(export)]
pub struct JobStatus {
    pub job_id: String,
    /// PENDING / RUNNING / COMPLETED / FAILED / ...
    pub state: String,
    pub partition: String,
    /// `squeue` 的 %M（已跑时长）或 sacct 的 Elapsed
    pub elapsed: String,
    /// 还在队列里 = true；已经归档 = false
    pub live: bool,
}

// ───────────────────────────── 校验 ─────────────────────────────
//
// 这些值最终会拼进远端 shell 脚本，所以**在拼之前**就把字符集卡死，
// 而不是靠引号去兜 —— 引号写错一次就是任意命令执行。

fn valid_case_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn valid_field(s: &str) -> bool {
    FIELDS.contains(&s)
}

fn valid_job_id(s: &str) -> bool {
    !s.is_empty() && s.len() <= 24 && s.chars().all(|c| c.is_ascii_digit() || c == '_')
}

// ───────────────────────────── 查作业状态 ─────────────────────────────

/// 只查**指定的**几个作业号，不是拉整个队列。
///
/// `squeue` 查活的，查不到再用 `sacct` 查归档 —— `MinJobAge=300` 意味着作业结束后
/// 还会在 squeue 里挂 5 分钟，所以两边都要看，squeue 优先。
fn status_script(ids: &[String]) -> String {
    let joined = ids.join(",");
    format!(
        r#"
echo '###LIVE'
squeue --jobs={joined} -h -o '%i|%T|%P|%M' 2>/dev/null
echo '###DONE'
sacct -j {joined} -X -P -n -o JobID,State,Partition,Elapsed 2>/dev/null
echo '###END'
"#
    )
}

pub fn parse_status(out: &str, want: &[String]) -> Vec<JobStatus> {
    let sections = sh::split_sections(out);
    let mut found: std::collections::HashMap<String, JobStatus> = Default::default();

    // sacct 先填，squeue 后填 —— 后者覆盖前者，因为它是当下的真相
    for (body, live) in [
        (sh::section(&sections, "DONE").unwrap_or(""), false),
        (sh::section(&sections, "LIVE").unwrap_or(""), true),
    ] {
        for line in body.lines() {
            let f: Vec<&str> = line.trim().split('|').collect();
            if f.len() < 4 || f[0].is_empty() {
                continue;
            }
            found.insert(
                f[0].to_string(),
                JobStatus {
                    job_id: f[0].to_string(),
                    // sacct 的 State 可能是 "CANCELLED by 12345"，只取头一个词
                    state: f[1].split_whitespace().next().unwrap_or(f[1]).to_string(),
                    partition: f[2].to_string(),
                    elapsed: f[3].to_string(),
                    live,
                },
            );
        }
    }
    // 按请求的顺序返回，查不到的给一个占位而不是消失 ——
    // 刚投出去的作业有几秒钟两边都查不到
    want.iter()
        .map(|id| {
            found.remove(id).unwrap_or_else(|| JobStatus {
                job_id: id.clone(),
                state: "UNKNOWN".into(),
                ..Default::default()
            })
        })
        .collect()
}

pub async fn job_status(ids: &[String]) -> Result<Vec<JobStatus>> {
    let ids: Vec<String> = ids.iter().filter(|s| valid_job_id(s)).cloned().collect();
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    if ids.len() > 16 {
        bail!("一次最多查 16 个作业");
    }
    let out = sh::run(&status_script(&ids), STATUS_TIMEOUT).await?;
    Ok(parse_status(&out, &ids))
}

// ───────────────────────────── 取视频 ─────────────────────────────

/// 产物在集群上的相对路径。两个阶段的命名规则不一样，这里是唯一的真源。
pub fn artifact_rel(stage: &str, case: &str, chunk: i32, field: &str) -> Result<String> {
    if !valid_case_name(case) {
        bail!("算例名不合法");
    }
    if !valid_field(field) {
        bail!("未知通道 {field}");
    }
    if !(0..=10).contains(&chunk) {
        bail!("chunk 超出范围");
    }
    Ok(match stage {
        "lift" => format!("lift/{case}/lift_chunk{chunk}_{field}_{STYLE}.mp4"),
        "lt" => format!("vis/{case}/longterm_chunk{chunk}_{field}_{STYLE}.mp4"),
        _ => bail!("stage 只能是 lift 或 lt"),
    })
}

/// 产物在盘上的绝对路径。
pub fn artifact_path(stage: &str, case: &str, chunk: i32, field: &str) -> Result<PathBuf> {
    let rel = artifact_rel(stage, case, chunk, field)?;
    Ok(PathBuf::from(repo_root()).join(crate::pipeline::WEB_ROOT).join(rel))
}

/// 读一个产物。文件不在就报错 —— 调用方据此回 404。
pub async fn read_artifact(stage: &str, case: &str, chunk: i32, field: &str) -> Result<Vec<u8>> {
    sh::read_file(&artifact_path(stage, case, chunk, field)?).await
}

#[cfg(test)]
mod tests {
    use super::*;






    #[test]
    fn 产物相对路径() {
        assert_eq!(
            artifact_rel("lift", "TK94", 9, "p_rgh").unwrap(),
            "lift/TK94/lift_chunk9_p_rgh_tri.mp4"
        );
        assert_eq!(
            artifact_rel("lt", "H0585_S350", 10, "alpha").unwrap(),
            "vis/H0585_S350/longterm_chunk10_alpha_tri.mp4"
        );
        assert!(artifact_rel("lift", "../etc", 9, "alpha").is_err());
        assert!(artifact_rel("prior", "TK94", 9, "alpha").is_err());
    }

    #[test]
    fn 产物绝对路径() {
        std::env::set_var("WAVE_ROOT", "/models");
        assert_eq!(
            artifact_path("lt", "TK94", 10, "alpha").unwrap().to_str().unwrap(),
            "/models/results/web/vis/TK94/longterm_chunk10_alpha_tri.mp4"
        );
        std::env::remove_var("WAVE_ROOT");
    }

    #[test]
    fn 作业状态_squeue_盖过_sacct() {
        // MinJobAge=300：作业刚结束时两边都有，squeue 才是当下的真相
        let out = "\
###LIVE
20937427|RUNNING|eecs|4:31
###DONE
20937427|RUNNING|eecs|00:04:31
20937398|COMPLETED|eecs|00:09:50
###END
";
        let want = vec!["20937427".to_string(), "20937398".to_string()];
        let s = parse_status(out, &want);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].job_id, "20937427");
        assert!(s[0].live);
        assert_eq!(s[0].elapsed, "4:31");
        assert!(!s[1].live);
        assert_eq!(s[1].state, "COMPLETED");
    }

    #[test]
    fn 作业状态_查不到也要占位() {
        // 刚 sbatch 完的几秒钟里 squeue 和 sacct 可能都还没有它
        let s = parse_status("###LIVE\n###DONE\n###END\n", &["999".to_string()]);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].state, "UNKNOWN");
    }

    #[test]
    fn 作业状态_取消原因不进状态() {
        let out = "###LIVE\n###DONE\n7|CANCELLED by 12345|eecs|00:01:00\n###END\n";
        let s = parse_status(out, &["7".to_string()]);
        assert_eq!(s[0].state, "CANCELLED");
    }
}
