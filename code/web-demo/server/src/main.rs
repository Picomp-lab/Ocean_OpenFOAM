//! 波浪场代理模型的交互式演示。
//!
//! 点一次按钮 → 在集群上按锁定的默认配置跑一遍 → 把渲好的视频放回页面。
//!
//! **跑在集群的登录节点上**：作业用 `bash -lc` 直接 sbatch，产物直接从盘上读，
//! 全程没有 ssh。前端由 rust-embed 烤进二进制，所以交付就是一个可执行文件。
//!
//! 状态全在 `results/web/` 一个目录里 —— 删掉 `priors/` 和 `vis/` 里的东西，
//! 下一次提交就会重算。

mod pipeline;
mod sh;
mod share;
mod wave;

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio::sync::Mutex;
use tracing::{info, warn};

/// 闲置多久自停（分钟）。0 = 不自停。
///
/// 这是跑在**共享登录节点**上的进程，不该留着没人管。前端只在有作业在跑时才
/// 轮询，且轮询请求带 `x-wave-poll` 头不算活动，所以一个忘了关的空标签页
/// 不会把服务吊住。
///
/// 作业跑着但没人看时也会停 —— 没关系，下次启动会跟集群对齐补上状态。
fn idle_timeout() -> Option<Duration> {
    let m: u64 = std::env::var("WAVE_IDLE_MIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);
    (m > 0).then(|| Duration::from_secs(m * 60))
}

pub fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

struct AppState {
    /// 提交记录。进程内是唯一真源，改一次就落一次盘。
    subs: Mutex<Vec<pipeline::Submission>>,
    /// 访客通道。None = 没共享。进程退出即关闭 —— 失效方向是「关」，不是「开」。
    guest: Mutex<Option<share::Guest>>,
    /// 正在跟集群对齐。同一时刻只允许一条 —— 页面开几个都只有一条 ssh。
    syncing: std::sync::atomic::AtomicBool,
    /// 上一次成功对齐的时刻，0 = 从没对齐过
    synced_at: std::sync::atomic::AtomicI64,
    /// 最后一次**真人操作**的时刻（unix 秒），驱动闲置自停
    last_seen: std::sync::atomic::AtomicI64,
    /// 闲置到点了从这里通知主循环退出
    quit: tokio::sync::Notify,
}

/// 请求是从哪条通道进来的。访客通道上的请求带 `Visitor(true)`。
#[derive(Clone, Copy)]
struct Visitor(bool);

// ---------- 接口 ----------

// ---------- 提交与流水线 ----------

/// 按锁定的默认配置提交一次计算。**不收任何参数** —— 这一版就是要它只能这么跑。
async fn submit(
    State(st): State<Arc<AppState>>,
    axum::Extension(visitor): axum::Extension<Visitor>,
) -> Result<Json<pipeline::Submission>, (axum::http::StatusCode, String)> {
    use axum::http::StatusCode;
    // 访客只能看。否则任何拿到链接的人都能用本人的账号往 SLURM 投作业、占用配额。
    if visitor.0 {
        return Err((
            StatusCode::FORBIDDEN,
            "共享链接是只读的 —— 提交计算要在本人的窗口里操作".into(),
        ));
    }
    let sub = pipeline::submit(&wave::repo_root())
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("{e:#}")))?;

    info!(
        "提交 {} · {} · 作业 {:?}",
        sub.id,
        sub.case,
        sub.job_ids()
    );

    let mut subs = st.subs.lock().await;
    subs.insert(0, sub.clone());
    // 记录只留最近 50 条，再多没人翻，json 也不该无限长
    subs.truncate(50);
    if let Err(e) = pipeline::save(&subs).await {
        warn!("提交记录落盘失败: {e:#}");
    }
    Ok(Json(sub))
}

/// 把还活着的记录跟集群对齐一次。
///
/// **ssh 不在锁里做** —— 先取作业号、放锁、查集群、再拿锁贴回去。
/// 一次查询最长 30 s，拿着锁等的话这段时间里所有请求都会卡住。
async fn sync_once(st: &AppState) {
    use std::sync::atomic::Ordering;

    let ids = {
        let subs = st.subs.lock().await;
        pipeline::live_job_ids(&subs)
    };
    if ids.is_empty() {
        return;
    }
    // 同一时刻只允许一条：开三个标签页不该变成三条 ssh
    if st.syncing.swap(true, Ordering::SeqCst) {
        return;
    }

    let rows = wave::job_status(&ids).await;
    st.syncing.store(false, Ordering::SeqCst);

    let rows = match rows {
        Ok(r) => r,
        // 查不到就沿用上次的状态，不清空、不判死 —— VPN 抖一下不该改写历史
        Err(e) => return warn!("刷新作业状态失败（沿用上次的）: {e:#}"),
    };
    st.synced_at.store(now(), Ordering::SeqCst);

    let mut subs = st.subs.lock().await;
    if pipeline::apply_status(&mut subs, rows, now()) {
        if let Err(e) = pipeline::save(&subs).await {
            warn!("提交记录落盘失败: {e:#}");
        }
    }
}

/// 两次对齐之间的最小间隔。页面多开或反复刷新不该变成对 slurmctld 的连击。
const SYNC_MIN_GAP: i64 = 5;

/// 提交记录列表。
///
/// **立刻返回，不等集群。** 记录本身来自本地的 `submissions.json`，跟 SLURM 无关；
/// 拿它去等一次可能几十秒才超时的 ssh，只会让页面在最该显示东西的时候空着。
///
/// 对齐丢后台跑，结果由下一次请求带回来（有作业在跑时前端每 15 秒问一次）。
/// `syncing` 让前端能把「正在刷新」显示出来。
async fn submissions(State(st): State<Arc<AppState>>) -> Json<pipeline::SubmissionList> {
    use std::sync::atomic::Ordering;

    let (items, has_live) = {
        let subs = st.subs.lock().await;
        (subs.clone(), !pipeline::live_job_ids(&subs).is_empty())
    };

    let synced = st.synced_at.load(Ordering::SeqCst);
    if has_live && now() - synced >= SYNC_MIN_GAP && !st.syncing.load(Ordering::SeqCst) {
        let st2 = st.clone();
        tokio::spawn(async move { sync_once(&st2).await });
    }

    Json(pipeline::SubmissionList {
        items,
        syncing: st.syncing.load(Ordering::SeqCst),
        synced_at: (synced > 0).then_some(synced),
    })
}

// ---------- 共享开关 ----------

#[derive(serde::Deserialize)]
struct ShareReq {
    on: bool,
}

/// 共享状态。`visitor` 让前端知道自己是不是从共享链接进来的
/// —— 是的话把提交按钮和开关都收起来。
async fn share_get(
    State(st): State<Arc<AppState>>,
    axum::Extension(visitor): axum::Extension<Visitor>,
) -> Json<serde_json::Value> {
    let g = st.guest.lock().await;
    Json(serde_json::json!({
        "on": g.is_some(),
        "addr": g.as_ref().map(|x| x.addr.clone()),
        "host": hostname(),
        "visitor": visitor.0,
        "idle_min": idle_timeout().map(|t| t.as_secs() / 60).unwrap_or(0),
    }))
}

async fn share_set(
    State(st): State<Arc<AppState>>,
    axum::Extension(visitor): axum::Extension<Visitor>,
    Json(req): Json<ShareReq>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    use axum::http::StatusCode;
    // 访客不能自己改开关，否则关了也能被别人开回来
    if visitor.0 {
        return Err((StatusCode::FORBIDDEN, "共享链接不能改共享设置".into()));
    }

    let mut g = st.guest.lock().await;
    match (req.on, g.take()) {
        // 开：已经开着就保持原样
        (true, Some(cur)) => {
            let addr = cur.addr.clone();
            *g = Some(cur);
            Ok(Json(serde_json::json!({ "on": true, "addr": addr, "host": hostname() })))
        }
        (true, None) => {
            let app = router(st.clone(), true);
            match share::open(app).await {
                Ok(new) => {
                    let addr = new.addr.clone();
                    info!("共享已打开 → {}:{}", hostname(), addr);
                    *g = Some(new);
                    Ok(Json(serde_json::json!({ "on": true, "addr": addr, "host": hostname() })))
                }
                Err(e) => Err((StatusCode::CONFLICT, format!("{e:#}"))),
            }
        }
        (false, cur) => {
            if let Some(c) = cur {
                c.stop();
                info!("共享已关闭");
            }
            Ok(Json(serde_json::json!({ "on": false, "addr": null, "host": hostname() })))
        }
    }
}

fn short_hostname() -> String {
    std::process::Command::new("hostname")
        .arg("-s")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "localhost".into())
}

/// 从**集群外面**能解析到这台机器的名字。
///
/// `hostname -s` 给的是 `submit-b`、`hostname -f` 给的是 `submit-b.ib.coehpc`
/// （InfiniBand 内部名），这两个在集群外都解析不了 —— 拿它拼出来的转发命令和
/// 共享链接都是不能用的。能用的是短名加上对外域名。
///
/// `WAVE_PUBLIC_HOST` 整个覆盖，`WAVE_PUBLIC_DOMAIN` 只换域名后缀。
fn hostname() -> String {
    if let Ok(h) = std::env::var("WAVE_PUBLIC_HOST") {
        return h;
    }
    let short = short_hostname();
    match std::env::var("WAVE_PUBLIC_DOMAIN") {
        Ok(d) if !d.is_empty() => format!("{short}.{d}"),
        _ => format!("{short}.hpc.engr.oregonstate.edu"),
    }
}

/// 锁定的配置。前端只读地照着它画那块说明，没有任何输入控件。
async fn config() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "run_name": pipeline::RUN_NAME,
        "run_ts": pipeline::RUN_TS,
        "case": pipeline::DEFAULT_CASE,
        "amp": pipeline::DEFAULT_AMP,
        "slope_label": pipeline::DEFAULT_SLOPE_LABEL,
        "chunk": pipeline::CHUNK,
        "fields": wave::FIELDS,
        "checkpoints": pipeline::scan_checkpoints(&wave::repo_root()),
    }))
}

/// 视频。产物就在本机盘上，直接读。
///
/// 支持 Range 是必须的：一段 1000 帧的视频 4–8 MB，没有 206 就拖不动进度条。
async fn video(
    Path((stage, case, chunk, field)): Path<(String, String, i32, String)>,
    headers: axum::http::HeaderMap,
) -> axum::response::Response {
    use axum::http::{header, StatusCode};
    use axum::response::IntoResponse;

    let data = match wave::read_artifact(&stage, &case, chunk, &field).await {
        Ok(d) => d,
        Err(e) => {
            warn!("取视频失败 {stage}/{case}/{chunk}/{field}: {e:#}");
            return (StatusCode::NOT_FOUND, format!("{e:#}")).into_response();
        }
    };
    let total = data.len() as u64;

    let range = headers
        .get(header::RANGE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| parse_range(v, total));

    let mut h = axum::http::HeaderMap::new();
    h.insert(header::CONTENT_TYPE, "video/mp4".parse().unwrap());
    h.insert(header::ACCEPT_RANGES, "bytes".parse().unwrap());
    // URL 不随重渲染变，所以只缓存一小会儿 —— FORCE=1 重跑后刷新就能看到新的
    h.insert(header::CACHE_CONTROL, "private, max-age=60".parse().unwrap());

    match range {
        Some((start, end)) => {
            let body = data[start as usize..=end as usize].to_vec();
            if let Ok(v) = format!("bytes {start}-{end}/{total}").parse() {
                h.insert(header::CONTENT_RANGE, v);
            }
            (StatusCode::PARTIAL_CONTENT, h, body).into_response()
        }
        None => (h, data).into_response(),
    }
}

/// 只认单段 `bytes=start-[end]`，多段 Range 浏览器播视频时不会发。
/// 返回闭区间 `[start, end]`；越界或反向一律当没给（退回 200 全量）。
fn parse_range(v: &str, total: u64) -> Option<(u64, u64)> {
    if total == 0 {
        return None;
    }
    let spec = v.trim().strip_prefix("bytes=")?;
    if spec.contains(',') {
        return None;
    }
    let (a, b) = spec.split_once('-')?;
    let (start, end) = match (a.trim(), b.trim()) {
        // "bytes=-500" = 最后 500 字节
        ("", suffix) => {
            let n: u64 = suffix.parse().ok()?;
            (total.saturating_sub(n.min(total)), total - 1)
        }
        (s, "") => (s.parse().ok()?, total - 1),
        (s, e) => (s.parse().ok()?, e.parse::<u64>().ok()?.min(total - 1)),
    };
    (start <= end && start < total).then_some((start, end))
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "repo_root": wave::repo_root(),
        "web_root": pipeline::WEB_ROOT,
        "store": pipeline::store_path().to_string_lossy(),
    }))
}

/// 前端产物在编译期烤进二进制，交付就是单个可执行文件。
#[derive(rust_embed::RustEmbed)]
#[folder = "../web/dist/"]
struct Assets;

async fn static_handler(uri: axum::http::Uri) -> axum::response::Response {
    use axum::http::{header, StatusCode};
    use axum::response::IntoResponse;

    let path = uri.path().trim_start_matches('/');

    // 打错的接口路径要 404，**不能**掉进下面的 SPA 回退拿到一页 HTML ——
    // 那样前端 fetch 收到的是 `<!doctype html>`，报出来是个莫名其妙的 JSON 解析错误
    if path.starts_with("api/") {
        return (StatusCode::NOT_FOUND, format!("没有这个接口: /{path}")).into_response();
    }

    let path = if path.is_empty() { "index.html" } else { path };

    // SPA 回退：其余未知路径一律给 index.html
    match Assets::get(path).or_else(|| Assets::get("index.html")) {
        Some(f) => (
            [(header::CONTENT_TYPE, f.metadata.mimetype().to_string())],
            f.data,
        )
            .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            "前端未构建 —— 先在 web/ 下跑 npm run build，再重新编译后端",
        )
            .into_response(),
    }
}

/// 后台轮询自己带的标记。带了就**不算活动** —— 见 `touch()`。
const POLL_HEADER: &str = "x-wave-poll";

/// 刷新「最后活动时刻」，但**只认真人动作**。
///
/// 前端在有作业跑的时候每 15 秒轮一次 `/api/submissions`，那是机器在动不是人在动。
/// 认它的话，一个忘了关的标签页会把服务一直续着，闲置自停就形同虚设。
/// 所以轮询请求带 `x-wave-poll: 1`，这里跳过；其余的都算：
/// 打开/刷新页面、点提交、开详情弹窗、拨共享开关，以及前端在真人
/// 点击/按键时发的 `/api/active`。
async fn touch(
    State(st): State<Arc<AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    if !req.headers().contains_key(POLL_HEADER) {
        st.last_seen
            .store(now(), std::sync::atomic::Ordering::Relaxed);
    }
    next.run(req).await
}

/// 真人动过了。前端在 pointerdown / keydown 时打这个（自己节流到每分钟最多一次）。
/// 本身什么都不做 —— `touch` 中间件已经把时间戳刷了。
async fn active() -> axum::http::StatusCode {
    axum::http::StatusCode::NO_CONTENT
}

/// 闲置看门狗。到点了通知主循环优雅退出（socket / pid 文件照常清理）。
async fn idle_watch(st: Arc<AppState>, timeout: Duration) {
    let secs = timeout.as_secs() as i64;
    loop {
        tokio::time::sleep(Duration::from_secs(60)).await;
        let idle = now() - st.last_seen.load(std::sync::atomic::Ordering::Relaxed);
        if idle >= secs {
            let shared = st.guest.lock().await.is_some();
            info!(
                "闲置 {} 分钟，自动停止{}",
                idle / 60,
                if shared { "（共享也一并关闭）" } else { "" }
            );
            st.quit.notify_waiters();
            return;
        }
    }
}

/// 两条通道共用同一套处理函数，只差一个 `Visitor` 标记。
fn router(st: Arc<AppState>, visitor: bool) -> Router {
    Router::new()
        .route("/api/config", get(config))
        .route("/api/submit", post(submit))
        .route("/api/submissions", get(submissions))
        .route("/api/share", get(share_get).post(share_set))
        .route("/api/active", post(active))
        .route("/api/video/{stage}/{case}/{chunk}/{field}", get(video))
        .route("/api/health", get(health))
        .fallback(get(static_handler))
        .layer(axum::middleware::from_fn_with_state(st.clone(), touch))
        .layer(axum::Extension(Visitor(visitor)))
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(st)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 日志同时进文件和 stdout。文件是给「事后查为什么没跑起来」用的 ——
    // 交互式启动时看 stdout，nohup 起来之后就只有文件了。
    let logp = pipeline::log_path();
    if let Some(d) = logp.parent() {
        std::fs::create_dir_all(d).ok();
    }
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "wave_demo=info".into());
    match std::fs::OpenOptions::new().create(true).append(true).open(&logp) {
        Ok(f) => {
            use tracing_subscriber::fmt::writer::MakeWriterExt;
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_ansi(false)
                .with_writer(std::sync::Arc::new(f).and(std::io::stdout))
                .init();
        }
        // 日志文件开不了不该拦着启动
        Err(e) => {
            tracing_subscriber::fmt().with_env_filter(filter).init();
            warn!("日志文件 {} 打不开（只写 stdout）: {e}", logp.display());
        }
    }
    info!("日志: {}", logp.display());

    let restored = pipeline::load().await;
    let n_live = pipeline::live_job_ids(&restored).len();
    if !restored.is_empty() {
        info!("载入 {} 条历史提交记录（{} 个作业还没结束）", restored.len(), n_live);
    }
    let st = Arc::new(AppState {
        subs: Mutex::new(restored),
        guest: Mutex::new(None),
        syncing: std::sync::atomic::AtomicBool::new(false),
        synced_at: std::sync::atomic::AtomicI64::new(0),
        last_seen: std::sync::atomic::AtomicI64::new(now()),
        quit: tokio::sync::Notify::new(),
    });

    let cks = pipeline::scan_checkpoints(&wave::repo_root());
    if cks.is_empty() {
        warn!(
            "{}/model/ 下没找到任何 .pt —— 提交会失败。见 README「首次部署」",
            pipeline::WEB_ROOT
        );
    } else {
        info!("checkpoint {} 个：", cks.len());
        for c in &cks {
            info!(
                "  {} {}/{}/{}  {:.1} MB",
                if c.current { "●" } else { "○" },
                c.run_name,
                c.run_ts,
                c.file,
                c.bytes as f64 / 1e6
            );
        }
    }

    match idle_timeout() {
        Some(t) => {
            info!("闲置 {} 分钟自动停止（WAVE_IDLE_MIN=0 可关掉）", t.as_secs() / 60);
            tokio::spawn(idle_watch(st.clone(), t));
        }
        None => info!("闲置自停已关闭"),
    }

    // 开机先跟集群对齐一次：进程停着的这段时间里作业可能已经跑完了，
    // 不补这一下的话，第一个打开页面的人会先看到一段过时的状态。
    // 连不上集群也不拦着启动 —— 页面还能显示上次的记录。
    if n_live > 0 {
        info!("正在跟集群对齐…");
        sync_once(&st).await;
        let subs = st.subs.lock().await;
        for s in subs.iter().take(n_live.min(8)) {
            info!("  {} · {} · {} {}", s.id, s.case, s.status, s.stage);
        }
    }

    let app = router(st.clone(), false);

    // pid 文件带主机名 —— 三台登录节点，光有 pid 不知道去哪台停
    let pidp = pipeline::pid_path();
    if let Some(prev) = std::fs::read_to_string(&pidp).ok() {
        warn!("发现旧的 pid 文件（上次没干净退出？）: {}", prev.trim());
    }
    // pid 文件里记**短名** —— stop.sh 拿 `hostname -s` 跟它比，写长名会对不上。
    // 对外展示用的长名是另一回事（见 hostname()）。
    std::fs::write(
        &pidp,
        format!("{} {}\n", std::process::id(), short_hostname()),
    )
    .ok();

    let sock = share::sock_path();
    let listener = share::bind_private()?;
    info!("监听 {}  (0600，只有本人能连)", sock.display());
    info!("  models = {}", wave::repo_root());
    info!(
        "  从自己机器上看： ssh -L 8788:{} {}@{}",
        sock.display(),
        std::env::var("USER").unwrap_or_else(|_| "<user>".into()),
        hostname()
    );

    let quit = st.clone();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let notified = quit.quit.notified();
            tokio::pin!(notified);
            // TERM 也要接：stop.sh / pkill 发的是 TERM，只接 ctrl-c 的话
            // socket 和 pid 文件会残留下来
            let mut term = tokio::signal::unix::signal(
                tokio::signal::unix::SignalKind::terminate(),
            )
            .expect("装 SIGTERM handler 失败");
            tokio::select! {
                _ = tokio::signal::ctrl_c() => info!("收到 ctrl-c"),
                _ = term.recv() => info!("收到 SIGTERM"),
                _ = &mut notified => {}        // 闲置看门狗
            }
        })
        .await?;
    // socket 和 pid 都是文件，进程走了要自己收拾
    std::fs::remove_file(&sock).ok();
    std::fs::remove_file(&pidp).ok();
    info!("已退出");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_range;

    #[test]
    fn range_单段() {
        assert_eq!(parse_range("bytes=0-499", 1000), Some((0, 499)));
        // 缺尾 = 到文件末尾；浏览器拖进度条发的就是这种
        assert_eq!(parse_range("bytes=500-", 1000), Some((500, 999)));
        // 尾部越界要夹到最后一个字节，不能原样返回
        assert_eq!(parse_range("bytes=0-99999", 1000), Some((0, 999)));
    }

    #[test]
    fn range_后缀式取末尾() {
        assert_eq!(parse_range("bytes=-500", 1000), Some((500, 999)));
        // 要的比文件还长 → 整个文件
        assert_eq!(parse_range("bytes=-5000", 1000), Some((0, 999)));
    }

    #[test]
    fn range_不合法一律退回全量() {
        // 起点越界、反向、多段、格式不对，都返回 None 让调用方走 200
        assert_eq!(parse_range("bytes=1000-1200", 1000), None);
        assert_eq!(parse_range("bytes=800-400", 1000), None);
        assert_eq!(parse_range("bytes=0-10,20-30", 1000), None);
        assert_eq!(parse_range("items=0-10", 1000), None);
        assert_eq!(parse_range("bytes=abc-def", 1000), None);
        assert_eq!(parse_range("bytes=0-499", 0), None);
    }
}
