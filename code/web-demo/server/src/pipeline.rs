//! 提交记录与流水线。
//!
//! **这一版只按锁定的默认配置跑，没有任何可调参数。** 一次提交 = 一条
//! `Submission`，底下挂两个 SLURM 作业，用 `afterok` 串起来：
//!
//! ```text
//!   (funwave)  →  prior  →  vis
//!    复用已有      5-6min    1h38m        （实测）
//!                  eecs      preempt,ampere
//! ```
//!
//! `vis` 这一段就是「HPM rollout + 渲染」—— `vis.py lt` 会加载 checkpoint 做完
//! 1000 帧 rollout 再出图，**不是**两个作业。
//!
//! FUNWAVE 那一段不跑：默认配置是训练工况 TK94，它的输出集群上已经有了。
//! 要放开参数（任意波高 → 现建算例 → 跑完整三段）的话，改的是这里，
//! 前端只是照着 `/api/config` 画。

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;
use ts_rs::TS;

use crate::{sh, wave};

const SUBMIT_TIMEOUT: Duration = Duration::from_secs(60);

/// 锁定的默认配置。前端不给选，后端也只认这一套。
pub const RUN_NAME: &str = "hpm_fw_aU_h128";
pub const RUN_TS: &str = "2026-08-12_15-31-45";
/// `lt` 只接了 chunk 10 —— 别的 chunk 要先标 t-offset，那条链路没接。
pub const CHUNK: i32 = 10;

/// demo 自己的输出根，与实验产物 `results/fwv/` 分开。
/// **删掉 `priors/` 或 `vis/` 里的东西，下一次提交就会重算** —— 演示"链路真的在跑"
/// 靠的就是这个。`model/` 是权重，删了整条链路起不来。
pub const WEB_ROOT: &str = "results/web";

/// 默认配置就是训练工况本身：TK94 = Ting & Kirby (1994) 溢波破碎。
/// 它的 FUNWAVE 输出集群上已经有了，所以第一段永远是复用。
pub const DEFAULT_CASE: &str = "TK94";
/// 造波机振幅 AMP_WK（m）。波高 H = 2·amp = 12.7 cm
pub const DEFAULT_AMP: f64 = 0.0635;
pub const DEFAULT_SLOPE: f64 = 0.028571429;
pub const DEFAULT_SLOPE_LABEL: &str = "1:35";

// ───────────────────────────── 数据结构 ─────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct StageRun {
    /// funwave | prior | vis
    pub kind: String,
    pub job_id: String,
    pub partition: String,
    /// SLURM 状态，由轮询更新；刚提交时是 UNKNOWN
    pub state: String,
    pub elapsed: String,
}

/// `/api/submissions` 的返回。
///
/// **记录本身来自本地的 `submissions.json`，不依赖集群** —— 所以它总是立刻返回。
/// 跟集群对齐是后台在做的事，`syncing` 让前端能把这件事显示出来，而不是
/// 拿整个列表去等一次可能很慢的 ssh。
#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct SubmissionList {
    pub items: Vec<Submission>,
    /// 正在跟集群对齐
    pub syncing: bool,
    /// 上一次成功对齐的时刻；从没对齐过是 None
    #[ts(as = "Option<i32>")]
    pub synced_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct Submission {
    pub id: String,
    #[ts(as = "i32")]
    pub created_at: i64,
    /// 造波机振幅 AMP_WK（m）。波高 H = 2·amp
    pub amp: f64,
    pub slope: f64,
    pub slope_label: String,
    /// 算例目录名
    pub case: String,
    /// true = 命中已有算例，FUNWAVE 那段跳过了
    pub reused_funwave: bool,
    pub chunk: i32,
    pub fields: Vec<String>,
    pub stages: Vec<StageRun>,
    /// queued | running | done | failed | cancelled
    pub status: String,
    /// 当前跑到哪一段：funwave | prior | vis | ""
    pub stage: String,
    /// 失败时的说明
    pub note: Option<String>,
}

impl Submission {
    pub fn live(&self) -> bool {
        self.status == "queued" || self.status == "running"
    }
    pub fn job_ids(&self) -> Vec<String> {
        self.stages.iter().map(|s| s.job_id.clone()).collect()
    }
}

/// 短 id，够区分就行，不追求全局唯一
fn new_id(created_at: i64) -> String {
    format!("{:x}", (created_at as u64).wrapping_mul(2654435761) & 0xffffff)
}

// ───────────────────────────── 扫 checkpoint ─────────────────────────────

/// `results/web/model/` 下的一个 checkpoint。
#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct Checkpoint {
    /// run 名，如 `hpm_fw_aU_h128`
    pub run_name: String,
    /// 时间戳目录，如 `2026-08-12_15-31-45`
    pub run_ts: String,
    /// 权重文件名，如 `best.pt`
    pub file: String,
    #[ts(as = "i32")]
    pub bytes: i64,
    /// 是不是当前锁定在用的那个
    pub current: bool,
}

/// 扫 `results/web/model/` 看有哪些权重可选。**每次启动扫一遍**，不写死在前端。
///
/// 认的布局是 `<web>/model/<run_name>/<ts>/checkpoints/*.pt` —— 跟集群上
/// `results/train/` 的结构一致，把训练产出的目录整个拷过来就能被认出来。
pub fn scan_checkpoints(root: &str) -> Vec<Checkpoint> {
    let base = PathBuf::from(root).join(WEB_ROOT).join("model");
    let mut out = Vec::new();

    let Ok(runs) = std::fs::read_dir(&base) else {
        return out;
    };
    for run in runs.flatten().filter(|e| e.path().is_dir()) {
        let run_name = run.file_name().to_string_lossy().into_owned();
        let Ok(stamps) = std::fs::read_dir(run.path()) else {
            continue;
        };
        for ts in stamps.flatten().filter(|e| e.path().is_dir()) {
            let run_ts = ts.file_name().to_string_lossy().into_owned();
            let Ok(files) = std::fs::read_dir(ts.path().join("checkpoints")) else {
                continue;
            };
            for f in files.flatten() {
                let p = f.path();
                if p.extension().and_then(|e| e.to_str()) != Some("pt") {
                    continue;
                }
                let file = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
                out.push(Checkpoint {
                    current: run_name == RUN_NAME && run_ts == RUN_TS && file == "best.pt",
                    run_name: run_name.clone(),
                    run_ts: run_ts.clone(),
                    file,
                    bytes: f.metadata().map(|m| m.len() as i64).unwrap_or(0),
                });
            }
        }
    }
    // 当前在用的排最前，其余按 run 名 + 时间戳，顺序稳定才好扫
    out.sort_by(|a, b| {
        b.current
            .cmp(&a.current)
            .then(a.run_name.cmp(&b.run_name))
            .then(a.run_ts.cmp(&b.run_ts))
            .then(a.file.cmp(&b.file))
    });
    out
}

// ───────────────────────────── 提交 ─────────────────────────────

/// 两段作业：抬升成 3D 先验 → HPM rollout + 渲染。
///
/// **没有第一段。** 默认配置就是训练工况 TK94，它的 FUNWAVE 输出集群上早就有了，
/// 所以链路永远从 prior 起跑。（前端仍然把 FUNWAVE 画成第一步并标「复用已有」，
/// 因为那确实是链路的一环，只是不用重跑。）
///
/// 两段放在一次 ssh 里提交，是因为**依赖关系必须原子地建立** —— 分两次投的话，
/// 第二次失败会留下一个没人管的孤儿 prior 作业。
///
/// 输出逐行 `kind|jobid|partition`。
fn pipeline_script(root: &str, case: &str, fields: &str) -> String {
    format!(
        r#"
set -euo pipefail
cd {root}/code
mkdir -p logs
WEBROOT="{root}/{WEB_ROOT}"

# ── 抬升成 3D 先验 ──
# PRIORROOT/VISROOT/CKDIR 把产物从实验目录 results/fwv/ 挪到 demo 自己的
# results/web/ 下（vis_adp.sh 里是 ${{VAR:-默认}}，不传就是原来的行为）。
# FORCE=1：被抢占重排时不能把半截的产物当成「已存在」跳过。
PJOB=$(STAGE=prior CHUNK={CHUNK} CASES='{case}' FORCE=1 \
  PRIORROOT="$WEBROOT/priors" \
  sbatch --parsable --job-name=wd_prior \
         --partition=eecs --gres=none --cpus-per-task=4 --mem=32G --time=04:00:00 \
         vis_adp.sh)
echo "prior|$PJOB|eecs"

# ── HPM rollout + 渲染 ──
# preempt 几乎不排队但可能被抢占，--requeue 让它自动重来；ampere 作为兜底，
# SLURM 会取两个分区里最先能跑的那个。
VJOB=$(STAGE=vis CHUNK={CHUNK} CASES='{case}' FIELDS='{fields}' STYLE=tri FORCE=1 \
  RUN_TS={RUN_TS} \
  PRIORROOT="$WEBROOT/priors" VISROOT="$WEBROOT/vis" \
  CKDIR="$WEBROOT/model/{RUN_NAME}/{RUN_TS}" \
  sbatch --parsable --job-name=wd_hpm --dependency=afterok:$PJOB --requeue \
         --partition=preempt,ampere --gres=gpu:1 --cpus-per-task=4 --mem=48G \
         --time=06:00:00 \
         vis_adp.sh)
echo "vis|$VJOB|preempt,ampere"
"#
    )
}

/// 按锁定的默认配置跑一次。没有参数 —— 这是刻意的，见模块头。
pub async fn submit(root: &str) -> Result<Submission> {
    let fields = wave::FIELDS.join(" ");
    let out = sh::run(&pipeline_script(root, DEFAULT_CASE, &fields), SUBMIT_TIMEOUT).await?;

    let mut stages = Vec::new();
    for line in out.lines() {
        let f: Vec<&str> = line.trim().split('|').collect();
        match f.as_slice() {
            ["ERR", msg] => bail!("集群上提交失败：{msg}"),
            [kind, id, part] if !id.is_empty() && id.chars().all(|c| c.is_ascii_digit()) => {
                stages.push(StageRun {
                    kind: kind.to_string(),
                    job_id: id.to_string(),
                    partition: part.to_string(),
                    state: "UNKNOWN".into(),
                    elapsed: String::new(),
                })
            }
            _ => {}
        }
    }
    if stages.len() != 2 {
        bail!("期望拿到 prior/vis 两个作业号，实际是：{}", out.trim());
    }

    let created_at = crate::now();
    Ok(Submission {
        id: new_id(created_at),
        created_at,
        amp: DEFAULT_AMP,
        slope: DEFAULT_SLOPE,
        slope_label: DEFAULT_SLOPE_LABEL.into(),
        case: DEFAULT_CASE.into(),
        reused_funwave: true,
        chunk: CHUNK,
        fields: wave::FIELDS.iter().map(|s| s.to_string()).collect(),
        stages,
        status: "queued".into(),
        stage: String::new(),
        note: None,
    })
}

// ───────────────────────────── 状态归并 ─────────────────────────────

fn is_bad(state: &str) -> bool {
    matches!(
        state,
        "FAILED" | "TIMEOUT" | "OUT_OF_MEMORY" | "NODE_FAIL" | "BOOT_FAIL" | "DEADLINE"
    )
}

/// 由三段的 SLURM 状态推出这次提交的总状态。
///
/// 顺序很重要：先看有没有炸，再看是不是全完了，最后才是「跑到哪一段」——
/// 反过来的话，第一段完成、第二段炸了会被报成「正在跑第二段」。
pub fn derive(sub: &mut Submission) {
    if sub.stages.iter().any(|s| is_bad(&s.state)) {
        sub.status = "failed".into();
        sub.stage = sub
            .stages
            .iter()
            .find(|s| is_bad(&s.state))
            .map(|s| s.kind.clone())
            .unwrap_or_default();
        return;
    }
    // CANCELLED 单独一档：被抢占重排的作业在 sacct 里也可能短暂显示 CANCELLED，
    // 所以只在没有任何一段还活着的时候才判定为「已取消」
    let any_live = sub
        .stages
        .iter()
        .any(|s| matches!(s.state.as_str(), "PENDING" | "RUNNING" | "COMPLETING" | "REQUEUED" | "UNKNOWN"));
    if !any_live && sub.stages.iter().any(|s| s.state == "CANCELLED") {
        sub.status = "cancelled".into();
        return;
    }
    if sub.stages.iter().all(|s| s.state == "COMPLETED") {
        sub.status = "done".into();
        sub.stage = String::new();
        return;
    }
    let cur = sub.stages.iter().find(|s| s.state != "COMPLETED");
    match cur {
        Some(s) => {
            sub.stage = s.kind.clone();
            sub.status = if s.state == "RUNNING" || s.state == "COMPLETING" {
                "running".into()
            } else {
                "queued".into()
            };
        }
        None => sub.status = "done".into(),
    }
}

// ───────────────────────────── 与集群对齐 ─────────────────────────────

/// 作业在 squeue 和 sacct 里都查不到时的宽限期。
///
/// 刚 sbatch 完的几秒钟两边都还没有它，所以不能一查不到就判死。过了这个时间
/// 还是查不到，就是真的没了 —— 要么当初没提交成功，要么记录比 sacct 的保留期还老。
const LOST_GRACE: i64 = 300;

/// 还活着的记录底下挂的作业号。查集群只查这些 —— 已经 done/failed 的不再问，
/// 否则历史越长每次越费 ssh。
pub fn live_job_ids(subs: &[Submission]) -> Vec<String> {
    subs.iter().filter(|s| s.live()).flat_map(|s| s.job_ids()).collect()
}

/// 把查回来的作业状态贴回记录上，重新归并总状态。返回是否有变化（有才需要落盘）。
///
/// 刻意做成**纯函数**：ssh 在外面做完再进来，这样锁不用跨 await 持有，
/// 也方便单测「作业丢了」这条路径。
pub fn apply_status(subs: &mut [Submission], rows: Vec<wave::JobStatus>, now: i64) -> bool {
    let by_id: std::collections::HashMap<String, wave::JobStatus> =
        rows.into_iter().map(|r| (r.job_id.clone(), r)).collect();
    let mut changed = false;

    for s in subs.iter_mut().filter(|s| s.live()) {
        let before = (s.status.clone(), s.stage.clone());
        for st in s.stages.iter_mut() {
            if let Some(r) = by_id.get(&st.job_id) {
                if st.state != r.state || st.elapsed != r.elapsed {
                    st.state = r.state.clone();
                    st.elapsed = r.elapsed.clone();
                    changed = true;
                }
            }
        }
        derive(s);

        // 两边都查不到，且已经过了宽限期 —— 别再等了。不然这条记录会永远停在
        // 「排队中」，前端也会为它一直轮询下去。
        if s.live()
            && s.stages.iter().any(|x| x.state == "UNKNOWN")
            && now - s.created_at > LOST_GRACE
        {
            s.status = "failed".into();
            s.note = Some("作业在集群上查不到了 —— 可能当初没提交成功，或已超出 sacct 的保留期".into());
            changed = true;
        }

        if (s.status.clone(), s.stage.clone()) != before {
            tracing::info!("{} · {} → {} {}", s.id, s.case, s.status, s.stage);
        }
    }
    changed
}

// ───────────────────────────── 落盘 ─────────────────────────────

/// demo 的状态目录 —— 提交记录、日志、pid 都在这儿，跟产物放一起。
fn state_dir() -> PathBuf {
    PathBuf::from(wave::repo_root()).join(WEB_ROOT)
}

/// 后端自己的日志。**不要跟 SLURM 作业的日志混淆** —— 那些在 `code/logs/` 下，
/// 记的是作业里发生了什么；这份记的是后端做了什么（提交时 sbatch 回了什么、
/// 跟集群对齐失败的原因）。
pub fn log_path() -> PathBuf {
    std::env::var("WAVE_LOG")
        .map(PathBuf::from)
        .unwrap_or_else(|_| state_dir().join("wave-demo.log"))
}

/// pid 文件，内容是 `<pid> <hostname>`。
///
/// **主机名是必需的**：submit 轮询解析到三台登录节点，光有 pid 不知道该去哪台停。
pub fn pid_path() -> PathBuf {
    std::env::var("WAVE_PID")
        .map(PathBuf::from)
        .unwrap_or_else(|_| state_dir().join("wave-demo.pid"))
}

pub fn store_path() -> PathBuf {
    if let Ok(p) = std::env::var("WAVE_STORE") {
        return PathBuf::from(p);
    }
    // 跟产物放一起：results/web/ 一个目录就是这个 demo 的全部状态
    state_dir().join("submissions.json")
}

pub async fn load() -> Vec<Submission> {
    let p = store_path();
    match tokio::fs::read(&p).await {
        Ok(b) => serde_json::from_slice(&b).unwrap_or_else(|e| {
            tracing::warn!("提交记录解析失败（当成空的继续）: {e}");
            Vec::new()
        }),
        Err(_) => Vec::new(),
    }
}

/// 先写临时文件再 rename —— 中途崩掉不会留下一个半截的 json
/// 把整份记录都毁掉。
pub async fn save(subs: &[Submission]) -> Result<()> {
    let p = store_path();
    if let Some(d) = p.parent() {
        tokio::fs::create_dir_all(d).await.ok();
    }
    let body = serde_json::to_vec_pretty(subs).context("序列化提交记录失败")?;
    let tmp = p.with_extension("json.tmp");
    tokio::fs::write(&tmp, &body).await?;
    tokio::fs::rename(&tmp, &p).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 扫_checkpoint_只认_pt_并把当前那个排最前() {
        let d = std::env::temp_dir().join(format!("wave-ck-{}", std::process::id()));
        let mk = |run: &str, ts: &str, f: &str| {
            let p = d.join(WEB_ROOT).join("model").join(run).join(ts).join("checkpoints");
            std::fs::create_dir_all(&p).unwrap();
            std::fs::write(p.join(f), b"x").unwrap();
        };
        mk("aaa_first_alphabetically", "2026-01-01_00-00-00", "best.pt");
        mk(RUN_NAME, RUN_TS, "best.pt");
        mk(RUN_NAME, RUN_TS, "latest.pt");
        mk(RUN_NAME, RUN_TS, "notes.txt"); // 不是 .pt，不该出现

        let got = scan_checkpoints(d.to_str().unwrap());
        assert_eq!(got.len(), 3, "只认 .pt：{got:?}");
        // 当前在用的排第一，哪怕它的 run 名字典序靠后
        assert!(got[0].current);
        assert_eq!((got[0].run_name.as_str(), got[0].file.as_str()), (RUN_NAME, "best.pt"));
        assert!(got[1..].iter().all(|c| !c.current));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn 目录不存在时返回空而不是崩() {
        assert!(scan_checkpoints("/definitely/not/here").is_empty());
    }




    fn sub_with(states: &[(&str, &str)]) -> Submission {
        Submission {
            id: "x".into(),
            created_at: 0,
            amp: 0.06,
            slope: DEFAULT_SLOPE,
            slope_label: "1:35".into(),
            case: "H0600_S350".into(),
            reused_funwave: false,
            chunk: CHUNK,
            fields: vec![],
            stages: states
                .iter()
                .map(|(k, st)| StageRun {
                    kind: k.to_string(),
                    job_id: "1".into(),
                    partition: "p".into(),
                    state: st.to_string(),
                    elapsed: String::new(),
                })
                .collect(),
            status: String::new(),
            stage: String::new(),
            note: None,
        }
    }

    #[test]
    fn 总状态跟着当前那一段走() {
        let mut s = sub_with(&[("funwave", "COMPLETED"), ("prior", "RUNNING"), ("vis", "PENDING")]);
        derive(&mut s);
        assert_eq!((s.status.as_str(), s.stage.as_str()), ("running", "prior"));

        let mut s = sub_with(&[("prior", "PENDING"), ("vis", "PENDING")]);
        derive(&mut s);
        assert_eq!((s.status.as_str(), s.stage.as_str()), ("queued", "prior"));

        let mut s = sub_with(&[("prior", "COMPLETED"), ("vis", "COMPLETED")]);
        derive(&mut s);
        assert_eq!(s.status, "done");
    }

    #[test]
    fn 后面炸了不能报成正在跑() {
        // 第一段完成、第二段失败 —— 必须是 failed 而不是「跑到 vis 了」
        let mut s = sub_with(&[
            ("funwave", "COMPLETED"),
            ("prior", "FAILED"),
            ("vis", "PENDING"),
        ]);
        derive(&mut s);
        assert_eq!((s.status.as_str(), s.stage.as_str()), ("failed", "prior"));
    }

    #[test]
    fn 查不到的作业过了宽限期判死() {
        let mut v = vec![sub_with(&[("prior", "UNKNOWN"), ("vis", "UNKNOWN")])];
        v[0].created_at = 1000;
        derive(&mut v[0]);
        assert!(v[0].live(), "刚提交时应该还算活着");

        // 宽限期内查不到：继续等
        assert!(!apply_status(&mut v, vec![], 1000 + LOST_GRACE - 1));
        assert!(v[0].live());

        // 过了宽限期还查不到：判死，并且带上说明
        assert!(apply_status(&mut v, vec![], 1000 + LOST_GRACE + 1));
        assert_eq!(v[0].status, "failed");
        assert!(v[0].note.is_some());
    }

    #[test]
    fn 贴回状态后总状态跟着走() {
        let mut v = vec![sub_with(&[("prior", "RUNNING"), ("vis", "PENDING")])];
        derive(&mut v[0]);
        let rows = vec![
            wave::JobStatus { job_id: "1".into(), state: "COMPLETED".into(), partition: "eecs".into(), elapsed: "00:05:12".into(), live: false },
        ];
        // 两段的 job_id 在 sub_with 里都是 "1"，所以两段都会被更新成 COMPLETED
        assert!(apply_status(&mut v, rows, 2000));
        assert_eq!(v[0].status, "done");
        assert_eq!(v[0].stages[0].elapsed, "00:05:12");
    }

    #[test]
    fn 已经结束的记录不再去查集群() {
        let mut v = vec![sub_with(&[("prior", "COMPLETED"), ("vis", "COMPLETED")])];
        derive(&mut v[0]);
        assert_eq!(v[0].status, "done");
        assert!(live_job_ids(&v).is_empty());
    }

    #[test]
    fn 抢占重排期间不算已取消() {
        // preempt 上被抢占时会短暂出现 CANCELLED，但只要还有段活着就不是终态
        let mut s = sub_with(&[("prior", "COMPLETED"), ("vis", "CANCELLED")]);
        derive(&mut s);
        assert_eq!(s.status, "cancelled");

        let mut s = sub_with(&[("prior", "CANCELLED"), ("vis", "PENDING")]);
        derive(&mut s);
        assert_eq!(s.status, "queued");
    }
}
