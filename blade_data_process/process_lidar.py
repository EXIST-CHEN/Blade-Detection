from pathlib import Path
import open3d as o3d
import numpy as np

# 定义路径
source_dir = Path('.vscode/raw_blade_data/pointcloud/rslidar')
velodyne_dir = Path('.vscode/raw_blade_data/processed/velodyne')
velodyne_reduced_dir = Path('.vscode/raw_blade_data/processed/velodyne_reduced')
calib_dir = Path('.vscode/raw_blade_data/processed/calib')

def clear_directory(directory: Path):
    if directory.exists():
        for item in directory.glob('*'):
            if item.is_file():
                item.unlink()  # 删除文件

# 创建目标目录
velodyne_dir.mkdir(parents=True, exist_ok=True)
clear_directory(velodyne_dir)
velodyne_reduced_dir.mkdir(parents=True, exist_ok=True)
clear_directory(velodyne_reduced_dir)
calib_dir.mkdir(parents=True, exist_ok=True)
clear_directory(calib_dir)

# 设置降采样参数
voxel_size = 0.1  # 可根据实际点云密度调整

# 遍历所有 .pcd 文件
for pcd_file in source_dir.glob('*.pcd'):
    if pcd_file.is_file():
        print(f"处理文件：{pcd_file.name}")

        # 读取点云
        pcd = o3d.io.read_point_cloud(str(pcd_file))

        # 移除无效点（NaN / Inf）
        pcd.remove_non_finite_points()

         # 原始点云保存为 BIN（强制 shape (N, 4)）
        points_xyz = np.asarray(pcd.points)
        # 添加强度列（全为 0）
        intensity = np.zeros((points_xyz.shape[0], 1), dtype=np.float32)
        points = np.hstack((points_xyz, intensity))  # 合并为 (N, 4)

         # 保存原始点云到 velodyne 目录
        bin_file_velodyne = velodyne_dir / (pcd_file.stem + '.bin')
        points.astype(np.float32).tofile(str(bin_file_velodyne))
        print(f"原始点数：{points.shape[0]}")
        print("合并前数组形状:", points.shape)  # 应为 (N, 4)
        print("合并前数据类型:", points.dtype)  # 应为 float32

        # 降采样处理
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)

        # 检查降采样后是否为空
        if len(pcd_downsampled.points) == 0:
            print(f"警告：{pcd_file.name} 降采样后无有效点，跳过保存。")
            continue

        # 仅保留 xyz 字段
        points_reduced = np.asarray(pcd_downsampled.points)
        
        # 添加 intensity 列（全为 0）
        intensity = np.zeros((points_reduced.shape[0], 1), dtype=np.float32)
        
        # 合并 xyz + intensity
        points_reduced = np.hstack((points_reduced, intensity))

        # 保存降采样后的点云到 velodyne_reduced 目录（xyz + intensity）
        bin_file_reduced = velodyne_reduced_dir / (pcd_file.stem + '.bin')
        points_reduced.astype(np.float32).tofile(str(bin_file_reduced))
        print(f"降采样后点数：{points_reduced.shape[0]}")
        print("降采样后数组形状:", points_reduced.shape)  # 应为 (N, 4)
        print("降采样后数据类型:", points_reduced.dtype)  # 应为 float32

        # 生成空的 calib 文件
        txt_file = calib_dir / (pcd_file.stem + '.txt')
        txt_file.touch(exist_ok=True)
