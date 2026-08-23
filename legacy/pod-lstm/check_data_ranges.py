"""
y=0.3 截面特征数据范围查询
============================
复用同目录下 pod_decomposition.py 的读取函数，
统计各变量在全部时间步上的全局 min/max/mean/std。

Usage:
    python check_data_ranges.py --data_dir /path/to/postProcessing/sample
    python check_data_ranges.py --data_dir ./sample --output ranges.txt --sample_every 10
"""

import os
import argparse
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy import stats


# ─────────────────────────────────────────────
# 从同目录的 pod_decomposition.py 导入读取函数
# ─────────────────────────────────────────────

def _import_pod_module():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pod_path = os.path.join(script_dir, "pod_decomposition.py")
    if not os.path.exists(pod_path):
        raise FileNotFoundError(
            f"找不到 pod_decomposition.py，请确保两个脚本在同一目录下。\n"
            f"查找路径: {pod_path}"
        )
    spec = importlib.util.spec_from_file_location("pod_decomposition", pod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

pod = _import_pod_module()
get_sorted_timesteps = pod.get_sorted_timesteps
read_raw_scalar      = pod.read_raw_scalar
read_raw_vector      = pod.read_raw_vector


# ─────────────────────────────────────────────
# 变量配置（与 pod_decomposition.py 保持一致）
# ─────────────────────────────────────────────

SCALAR_VARS = ["alpha.water", "p_rgh", "nut"]
VECTOR_VAR  = "U"
VECTOR_COMPONENTS = ["Ux", "Uy", "Uz"]


# ─────────────────────────────────────────────
# 核心：增量式范围累积
# ─────────────────────────────────────────────

class RangeAccumulator:
    """
    使用并行 Welford 算法 (Chan's algorithm) 进行稳定的增量均值/方差统计，
    避免直接累加平方和导致的大数相减精度丢失 (Catastrophic Cancellation)。
    """
    def __init__(self, name):
        self.name = name
        self.vmin = np.inf
        self.vmax = -np.inf
        self.count = 0
        self._mean = 0.0
        self._m2 = 0.0  # 离差平方和

    def update(self, arr: np.ndarray):
        arr_flat = arr.astype(np.float64).flatten()
        if arr_flat.size == 0:
            return

        self.vmin = min(self.vmin, arr_flat.min())
        self.vmax = max(self.vmax, arr_flat.max())

        n_b = arr_flat.size
        mean_b = arr_flat.mean()
        m2_b = ((arr_flat - mean_b) ** 2).sum()

        n_a = self.count
        mean_a = self._mean
        m2_a = self._m2

        n_ab = n_a + n_b
        delta = mean_b - mean_a

        mean_ab = mean_a + delta * n_b / n_ab
        m2_ab = m2_a + m2_b + (delta ** 2) * n_a * n_b / n_ab

        self.count = n_ab
        self._mean = mean_ab
        self._m2 = m2_ab

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        if self.count < 2:
            return 0.0
        return np.sqrt(self._m2 / self.count)

    def summary(self):
        return dict(min=self.vmin, max=self.vmax,
                    mean=self.mean, std=self.std, count=self.count)


def compute_ranges(data_dir, sample_every=1):
    timesteps = get_sorted_timesteps(data_dir)
    if not timesteps:
        raise RuntimeError(f"在 {data_dir} 下没有找到时间步目录")

    sampled = timesteps[::sample_every]
    print(f"共 {len(timesteps)} 个时间步，采样间隔={sample_every}，"
          f"实际处理 {len(sampled)} 步\n")

    accs = {v: RangeAccumulator(v) for v in SCALAR_VARS + VECTOR_COMPONENTS}
    n = len(sampled)

    for i, (t_val, t_name) in enumerate(sampled):
        if i % max(1, n // 10) == 0:
            print(f"  进度: {i+1}/{n}  (t={t_val:.2f}s)")

        t_dir = os.path.join(data_dir, t_name)

        for var in SCALAR_VARS:
            fpath = os.path.join(t_dir, f"{var}_ySlice.raw")
            if not os.path.exists(fpath):
                continue
            _, vals = read_raw_scalar(fpath)
            accs[var].update(vals)

        fpath = os.path.join(t_dir, f"{VECTOR_VAR}_ySlice.raw")
        if os.path.exists(fpath):
            _, ux, uy, uz = read_raw_vector(fpath)
            accs["Ux"].update(ux)
            accs["Uy"].update(uy)
            accs["Uz"].update(uz)

    return {name: acc.summary() for name, acc in accs.items() if acc.count > 0}


def collect_and_plot_one_var(var, is_vector_component, data_dir, sample_every, results, plot_dir):
    """Read data for a single variable, plot distribution, then release memory."""
    timesteps = get_sorted_timesteps(data_dir)
    sampled = timesteps[::sample_every]
    n = len(sampled)

    buf = []
    for i, (t_val, t_name) in enumerate(sampled):
        if i % max(1, n // 10) == 0:
            print(f"    [{var}] {i+1}/{n}  (t={t_val:.2f}s)")
        t_dir = os.path.join(data_dir, t_name)

        if is_vector_component:
            fpath = os.path.join(t_dir, f"{VECTOR_VAR}_ySlice.raw")
            if not os.path.exists(fpath):
                continue
            _, ux, uy, uz = read_raw_vector(fpath)
            comp = {"Ux": ux, "Uy": uy, "Uz": uz}[var]
            buf.append(comp)
        else:
            fpath = os.path.join(t_dir, f"{var}_ySlice.raw")
            if not os.path.exists(fpath):
                continue
            _, vals = read_raw_scalar(fpath)
            buf.append(vals)

    data = np.concatenate(buf).astype(np.float64)
    buf.clear()

    r = results[var]
    mu, sigma = r['mean'], r['std']

    lo = np.percentile(data, 0.05)
    hi = np.percentile(data, 99.95)
    data_clipped = data[(data >= lo) & (data <= hi)]

    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    # Label changed: '数据分布' -> 'Data Distribution'
    ax.hist(data_clipped, bins=120, density=True,
            color='steelblue', alpha=0.6, label='Data Distribution')

    try:
        stride = max(1, len(data_clipped) // 50000)
        kde = stats.gaussian_kde(data_clipped[::stride])
        x_kde = np.linspace(lo, hi, 500)
        # Label changed: 'KDE' (Already English)
        ax.plot(x_kde, kde(x_kde), 'steelblue', linewidth=2, label='KDE')
    except Exception:
        pass

    x_norm = np.linspace(lo, hi, 500)
    # Label changed: '正态' -> 'Normal'
    ax.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma),
            'r--', linewidth=1.5, label=f'Normal N({mu:.3g}, {sigma:.3g}²)')

    # Label changed: '均值' -> 'Mean'
    ax.axvline(mu, color='orange', linewidth=1.2, linestyle=':', label=f'Mean {mu:.3g}')

    total = r['count']
    # Title changed: '个样本' -> 'Samples'
    ax.set_title(f'{var} Distribution — {total/1e6:.1f}M Samples', fontsize=13, fontweight='bold')
    # Axes changed: '值' -> 'Value', '密度' -> 'Density'
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    safe_name = var.replace('.', '_')
    out = os.path.join(plot_dir, f"dist_{safe_name}.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')

    print(f"  Saved: {out}")

    del data, data_clipped


def plot_distributions(data_dir, sample_every, results, plot_dir, n_workers=6):
    """6个变量并发读取+画图，总内存峰值约 54GB，需申请 64G。"""
    os.makedirs(plot_dir, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    vector_components = set(VECTOR_COMPONENTS)
    vars_to_plot = [v for v in VAR_ORDER if v in results]

    def _task(var):
        collect_and_plot_one_var(
            var=var,
            is_vector_component=(var in vector_components),
            data_dir=data_dir,
            sample_every=sample_every,
            results=results,
            plot_dir=plot_dir,
        )
        return var

    print(f"  并发处理 {len(vars_to_plot)} 个变量（workers={n_workers}）")
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_task, var): var for var in vars_to_plot}
        for f in as_completed(futures):
            var = futures[f]
            try:
                f.result()
                print(f"  [完成] {var}")
            except Exception as e:
                print(f"  [ERROR] {var}: {e}")


# ─────────────────────────────────────────────
# 报告输出
# ─────────────────────────────────────────────

VAR_ORDER = ["alpha.water", "Ux", "Uy", "Uz", "p_rgh", "nut"]

NORM_HINTS = {
    "alpha.water": "值域 [0,1]，可直接用；或 MinMax",
    "Ux":          "z-score 推荐（主流方向）",
    "Uy":          "量级极小，注意数值稳定性",
    "Uz":          "z-score 推荐（垂向）",
    "p_rgh":       "量级大，推荐 z-score 或 / (rho*g*H)",
    "nut":         "右偏分布，建议 log1p 后再 z-score",
}


def print_report(results, n_sampled, output_path=None):
    col_w = 12
    header = (f"{'变量':<12}  {'min':>{col_w}}  {'max':>{col_w}}"
              f"  {'mean':>{col_w}}  {'std':>{col_w}}  {'样本数':>16}")
    sep = "─" * len(header)

    lines = [
        "=" * len(header),
        f"y=0.3 截面 — 特征数据范围统计  (采样时间步: {n_sampled})",
        sep, header, sep,
    ]

    for var in VAR_ORDER:
        if var not in results:
            continue
        r = results[var]
        lines.append(
            f"{var:<12}  {r['min']:>{col_w}.4g}  {r['max']:>{col_w}.4g}"
            f"  {r['mean']:>{col_w}.4g}  {r['std']:>{col_w}.4g}"
            f"  {r['count']:>16,}"
        )

    lines += [sep, "", "── 归一化建议 " + "─" * 40]
    for var in VAR_ORDER:
        if var in results:
            lines.append(f"  {var:<12}  {NORM_HINTS.get(var, '')}")
    lines.append("")

    report = "\n".join(lines)
    print(report)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"已保存到: {output_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="y=0.3 截面特征范围查询")
    parser.add_argument("--data_dir", required=True,
                        help="postProcessing/sample 目录路径")
    parser.add_argument("--output", default=None,
                        help="保存统计结果到此文本文件（可选）")
    parser.add_argument("--sample_every", type=int, default=1,
                        help="每隔 N 步采一次，快速预览用（默认=1 全部）")
    parser.add_argument("--plot", action="store_true",
                        help="同时输出数值分布图 distributions.png")
    args = parser.parse_args()

    results = compute_ranges(args.data_dir, sample_every=args.sample_every)
    timesteps = get_sorted_timesteps(args.data_dir)
    n_sampled = len(timesteps[::args.sample_every])

    print()
    print_report(results, n_sampled, output_path=args.output)

    if args.plot:
        plot_dir = os.path.dirname(os.path.abspath(args.output)) if args.output else "."
        print(f"\n开始逐变量输出分布图，保存到: {plot_dir}")
        plot_distributions(args.data_dir, args.sample_every, results, plot_dir)


if __name__ == "__main__":
    main()