//! 共享开关。
//!
//! 这个 app 跑在**共享的登录节点**上，那台机器上同时有十几个别的用户。关键在于：
//! `127.0.0.1` 是「这台机器」，不是「这个用户」—— TCP 端口没有 owner、没有 chmod，
//! 同机任何 UID 都能 connect 上来。所以「绑回环 = 私有」是错的。
//!
//! 因此分成两条通道：
//!
//! | | 走什么 | 谁能连 |
//! |---|---|---|
//! | 本人 | Unix socket，`0600` | **只有属主的 UID** —— 内核按文件权限拦，其他人 EACCES |
//! | 访客 | TCP，开关控制 | 开着时网络可达的人；关掉就整个不存在 |
//!
//! Unix socket 是操作系统级别的隔离，不是「端口猜不到」这种运气。
//!
//! 访客通道是**只读**的：`POST /api/submit` 会被挡掉。拿到链接的人能看结果，
//! 但不能用本人的账号往 SLURM 投作业、占用配额。

use anyhow::{Context, Result};
use std::path::PathBuf;
use tokio::sync::oneshot;

/// 自己走的那条通道，放在 `results/web/` 下 —— 这个 demo 读写的东西**全部**
/// 落在 `<models>` 里面，没有任何一样散在外面。`WAVE_SOCK` 可覆盖。
///
/// （`/nfs/hpc/share` 是 Lustre，实测能 bind unix socket。）
pub fn sock_path() -> PathBuf {
    if let Ok(p) = std::env::var("WAVE_SOCK") {
        return PathBuf::from(p);
    }
    PathBuf::from(crate::wave::repo_root())
        .join(crate::pipeline::WEB_ROOT)
        .join("wave-demo.sock")
}

/// 开共享时 TCP 监听在哪。默认所有网卡 —— 只监听回环的话，共享给别人就没意义了。
pub fn share_addr() -> String {
    std::env::var("WAVE_SHARE_ADDR").unwrap_or_else(|_| "0.0.0.0:8788".to_string())
}

/// 建 Unix socket 并把权限收到 0600。
///
/// 先删掉可能残留的旧 socket —— 进程被 kill -9 时不会清理，留着会 EADDRINUSE。
/// **权限要在 bind 之后立刻改**：bind 出来的默认权限受 umask 影响，可能是 0755。
pub fn bind_private() -> Result<tokio::net::UnixListener> {
    use std::os::unix::fs::PermissionsExt;

    let p = sock_path();
    if p.exists() {
        std::fs::remove_file(&p).ok();
    }
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).ok();
    }
    let l = tokio::net::UnixListener::bind(&p)
        .with_context(|| format!("bind {} 失败", p.display()))?;
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("给 {} 设 0600 失败", p.display()))?;
    Ok(l)
}

/// 正在开着的访客通道。drop 掉就等于关掉监听。
pub struct Guest {
    pub addr: String,
    stop: Option<oneshot::Sender<()>>,
}

impl Guest {
    pub fn stop(mut self) {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
    }
}

/// 起一个 TCP 监听跑访客路由。返回的 `Guest` 一旦 `stop()`，监听立刻关闭。
pub async fn open(app: axum::Router) -> Result<Guest> {
    let addr = share_addr();
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("bind {addr} 失败（端口被占？）"))?;
    let real = listener
        .local_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| addr.clone());

    let (tx, rx) = oneshot::channel();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = rx.await;
            })
            .await;
        tracing::info!("访客通道已关闭");
    });
    Ok(Guest { addr: real, stop: Some(tx) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[tokio::test]
    async fn 私有通道权限必须是_0600() {
        let p = std::env::temp_dir().join(format!("wave-test-{}.sock", std::process::id()));
        std::env::set_var("WAVE_SOCK", &p);
        let _l = bind_private().unwrap();
        let mode = std::fs::metadata(&p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "socket 权限是 {mode:o}，别的用户能连上");
        std::fs::remove_file(&p).ok();
        std::env::remove_var("WAVE_SOCK");
    }

    #[tokio::test]
    async fn 残留的旧_socket_不挡住启动() {
        let p = std::env::temp_dir().join(format!("wave-stale-{}.sock", std::process::id()));
        std::fs::write(&p, b"").unwrap(); // 模拟进程被强杀后残留的 socket 文件
        std::env::set_var("WAVE_SOCK", &p);
        assert!(bind_private().is_ok(), "残留文件应该被删掉而不是报 EADDRINUSE");
        std::fs::remove_file(&p).ok();
        std::env::remove_var("WAVE_SOCK");
    }
}
