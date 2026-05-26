import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

coords = np.load('../data/coords_2d.npy')
crop_mask = (coords[:,0] >= -2) & (coords[:,0] <= 16.45)
cc = coords[crop_mask]

# 用KDTree找每个点到最近邻的距离
tree = cKDTree(cc)
dists, _ = tree.query(cc, k=2)  # k=2: 自己+最近邻
nn_dist = dists[:, 1]  # 第二列是最近邻距离

print(f'Nearest neighbor distance:')
print(f'  min:    {nn_dist.min():.6f}m')
print(f'  median: {np.median(nn_dist):.6f}m')
print(f'  mean:   {nn_dist.mean():.6f}m')
print(f'  max:    {nn_dist.max():.6f}m')
print(f'  p5:     {np.percentile(nn_dist, 5):.6f}m')
print(f'  p25:    {np.percentile(nn_dist, 25):.6f}m')
print(f'  p75:    {np.percentile(nn_dist, 75):.6f}m')
print(f'  p95:    {np.percentile(nn_dist, 95):.6f}m')

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 直方图
axes[0].hist(nn_dist, bins=200, edgecolor='none')
axes[0].set_xlabel('Nearest neighbor distance (m)')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of point spacing')
axes[0].axvline(np.median(nn_dist), color='r', ls='--', label=f'median={np.median(nn_dist):.4f}m')
axes[0].legend()

# 空间分布：用颜色显示每个点的局部密度
sc = axes[1].scatter(cc[:,0], cc[:,1], c=nn_dist, s=0.3, cmap='viridis', vmax=np.percentile(nn_dist, 95))
plt.colorbar(sc, ax=axes[1], label='NN distance (m)')
axes[1].set_xlabel('x'); axes[1].set_ylabel('z')
axes[1].set_title('Local mesh spacing')

plt.tight_layout()
plt.savefig('mesh_spacing.png', dpi=150, bbox_inches='tight')
print('Saved mesh_spacing.png')