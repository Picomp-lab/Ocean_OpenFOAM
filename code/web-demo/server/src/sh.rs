//! 跑本机命令、读本机文件。
//!
//! 服务跑在集群的登录节点上，数据和作业都在本地，因此不需要任何远程调用。
//!
//! 命令一律包一层 `bash -ls`：
//!   `-l` 拿 login shell 的 PATH —— **非交互式 shell 的 PATH 里没有 SLURM 命令**
//!        （只有 `/usr/local/bin:/usr/bin:...`），直接 exec `sbatch` 会 command not found
//!   `-s` 从 stdin 读脚本，于是完全不需要处理引号转义

use anyhow::{bail, Context, Result};
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

/// 跑一段脚本，返回 stdout。超时会杀掉子进程。
pub async fn run(script: &str, timeout: Duration) -> Result<String> {
    let mut child = Command::new("bash")
        .args(["-ls"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .context("spawn bash 失败")?;

    let mut stdin = child.stdin.take().expect("stdin piped");
    let script = script.to_string();
    tokio::spawn(async move {
        let _ = stdin.write_all(script.as_bytes()).await;
        let _ = stdin.shutdown().await;
    });

    let out = match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(r) => r.context("等待 bash 退出失败")?,
        // child 在这里被 drop，kill_on_drop 收尸
        Err(_) => bail!("命令超时（{}s）", timeout.as_secs()),
    };
    if !out.status.success() {
        bail!(
            "退出码 {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
                .trim()
                .chars()
                .take(300)
                .collect::<String>()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// 产物就在本机盘上，直接读。
pub async fn read_file(path: &Path) -> Result<Vec<u8>> {
    let meta = tokio::fs::metadata(path)
        .await
        .with_context(|| format!("找不到 {}", path.display()))?;
    if meta.len() == 0 {
        bail!("{} 是空文件", path.display());
    }
    tokio::fs::read(path)
        .await
        .with_context(|| format!("读 {} 失败", path.display()))
}

/// 把 `###MARKER` 分隔的输出切成 (marker, body)，一次调用拿多段结果用。
pub fn split_sections(out: &str) -> Vec<(String, String)> {
    let mut sections: Vec<(String, String)> = Vec::new();
    for line in out.lines() {
        if let Some(name) = line.strip_prefix("###") {
            sections.push((name.trim().to_string(), String::new()));
        } else if let Some(last) = sections.last_mut() {
            last.1.push_str(line);
            last.1.push('\n');
        }
    }
    sections
}

pub fn section<'a>(sections: &'a [(String, String)], name: &str) -> Option<&'a str> {
    sections
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, b)| b.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 分段() {
        let s = split_sections("###A\n1\n2\n###B\nx\n###END\n");
        assert_eq!(section(&s, "A").unwrap(), "1\n2\n");
        assert_eq!(section(&s, "B").unwrap(), "x\n");
        assert_eq!(section(&s, "C"), None);
    }

    #[test]
    fn 标记之前的内容被丢掉() {
        // login shell 的 motd / 环境噪音会出现在第一个 ### 之前
        let s = split_sections("Welcome to the cluster\n###A\n1\n");
        assert_eq!(s.len(), 1);
        assert_eq!(section(&s, "A").unwrap(), "1\n");
    }
}
