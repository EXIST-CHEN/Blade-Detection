import os
import bisect
from pathlib import Path

# 数据源
src_dir = "/root/workspace/mmdetection3d/.vscode/raw_blade_data/processed"
src_image_dir = os.path.join(src_dir, "image_2")
src_label_dir = os.path.join(src_dir, "label_2")
src_velodyne_dir = os.path.join(src_dir, "velodyne")
src_velodyne_reduces_dir = os.path.join(src_dir, "velodyne_reduced")
src_calib_dir = os.path.join(src_dir, "calib")

# 目标目录
dst_base_dir = "/root/workspace/mmdetection3d/data/blade"
dst_training_dir = os.path.join(dst_base_dir, "training")
dst_training_dirs = {
    "image": os.path.join(dst_training_dir, "image_2"),
    "label": os.path.join(dst_training_dir, "label_2"),
    "velodyne": os.path.join(dst_training_dir, "velodyne"),
    "velodyne_reduced": os.path.join(dst_training_dir, "velodyne_reduced"),
    "calib": os.path.join(dst_training_dir, "calib")
}
dst_testing_dir = os.path.join(dst_base_dir, "testing")
dst_testing_dirs = {
    "image": os.path.join(dst_testing_dir, "image_2"),
    "velodyne": os.path.join(dst_testing_dir, "velodyne"),
    "velodyne_reduced": os.path.join(dst_testing_dir, "velodyne_reduced"),
    "calib": os.path.join(dst_testing_dir, "calib")
}
dst_imagesets_dir = os.path.join(dst_base_dir, "ImageSets")

# 清理已有链接
def clean_links(dir):
    # 用于统计删除的符号链接数量
    deleted_links = 0
    # 用于记录删除的符号链接路径
    deleted_paths = []
    # 递归遍历目录
    for dirpath, dirnames, filenames in os.walk(dir):
        for name in filenames + dirnames:
            full_path = os.path.join(dirpath, name)
            if os.path.islink(full_path):  # 判断是否为符号链接
                try:
                    os.unlink(full_path)  # 删除符号链接
                    deleted_links += 1
                    deleted_paths.append(full_path)
                    print(f"已删除符号链接: {full_path}")
                except Exception as e:
                    print(f"无法删除符号链接 {full_path}: {e}")
    # 输出统计信息
    print(f"共删除 {deleted_links} 个符号链接。")

# 创建输出目录（如果不存在）
def mkdirs(dirs):
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

# 提取时间戳函数
def get_timestamp_str(filename):
    base = os.path.splitext(filename)[0]
    return base  # 直接返回原始时间戳字符串

# 读取图片、图片标签、校准文件的时间戳映射
def build_timestamp_map(directory):
    timestamp_map = {}
    for f in os.listdir(directory):
        ts = get_timestamp_str(f)
        if ts is not None:
            timestamp_map[ts] = f
    return timestamp_map

# 查找与雷达时间最相近的图片
def find_matchs():
    # 构建时间戳列表和映射
    image_map = build_timestamp_map(src_image_dir)
    label_map = build_timestamp_map(src_label_dir)
    calib_map = build_timestamp_map(src_calib_dir)
    velo_red_map = build_timestamp_map(src_velodyne_reduces_dir)

    # 读取雷达文件并提取时间戳
    velodyne_files = []
    for f in os.listdir(src_velodyne_dir):
        ts = get_timestamp_str(f)
        if ts is not None:
            velodyne_files.append((ts, f))

    # 按时间戳排序
    velodyne_files.sort()
    image_times = sorted(image_map.keys())
    image_filenames = {ts: image_map[ts] for ts in image_times}

    # 存储匹配结果
    matches = []

    # 为每个雷达文件查找最接近的图片
    for velo_ts, velo_file in velodyne_files:
        idx = bisect.bisect_left(image_times, velo_ts)
        candidates = []

        if idx > 0:
            candidates.append(image_times[idx - 1])
        if idx < len(image_times):
            candidates.append(image_times[idx])

        if candidates is None:
            print(f"警告：未找到与雷达文件 {velo_file} 匹配的图片")
            continue

        # 找出最接近的时间戳
        def str_to_nsec(ts_str):
            sec, nsec = ts_str.split('.')
            return int(sec) * 1_000_000_000 + int(nsec)

        closest_ts = min(candidates, key=lambda x: abs(str_to_nsec(x) - str_to_nsec(velo_ts)))
        closest_image = image_filenames[closest_ts]
        closest_label = label_map.get(closest_ts)
        calib_file = calib_map.get(velo_ts)
        velo_red_file = velo_red_map.get(velo_ts)

        if not all([closest_image, closest_label, calib_file, velo_red_file]):
            print(f"警告：雷达文件 {velo_file} 缺少对应的图片、标签或校准文件")
            continue

        matches.append((velo_file, velo_red_file, closest_image, closest_label, calib_file))

    # 按雷达文件时间戳排序
    matches.sort(key=lambda x: get_timestamp_str(x[0]))
    return matches

# 建立统一的符号链接
def link_matches(matches):
    # 建立符号链接并重命名
    for i, (velo_file, velo_red_file, image_file, label_file, calib_file) in enumerate(matches):
        index = f"{i:06d}"

        # 构建源路径
        velo_src = os.path.join(src_velodyne_dir, velo_file)
        velo_red_src = os.path.join(src_velodyne_reduces_dir, velo_red_file)
        image_src = os.path.join(src_image_dir, image_file)
        label_src = os.path.join(src_label_dir, label_file)
        calib_src = os.path.join(src_calib_dir, calib_file)

        # 构建目标路径
        train_velo_dst = os.path.join(dst_training_dirs["velodyne"], f"{index}.bin")
        train_velo_red_dst = os.path.join(dst_training_dirs["velodyne_reduced"], f"{index}.bin")
        train_image_dst = os.path.join(dst_training_dirs["image"], f"{index}.png")
        train_label_dst = os.path.join(dst_training_dirs["label"], f"{index}.txt")
        train_calib_dst = os.path.join(dst_training_dirs["calib"], f"{index}.txt")
        test_velo_dst = os.path.join(dst_testing_dirs["velodyne"], f"{index}.bin")
        test_velo_red_dst = os.path.join(dst_testing_dirs["velodyne_reduced"], f"{index}.bin")
        test_image_dst = os.path.join(dst_testing_dirs["image"], f"{index}.png")
        test_calib_dst = os.path.join(dst_testing_dirs["calib"], f"{index}.txt")

        # 创建符号链接（如果已存在则跳过）
        for src, dst in [
            (velo_src, train_velo_dst),
            (velo_src, test_velo_dst),
            (velo_red_src, train_velo_red_dst),
            (velo_red_src, test_velo_red_dst),
            (calib_src, train_calib_dst),
            (calib_src, test_calib_dst),
            (image_src, train_image_dst),
            (image_src, test_image_dst),
            (label_src, train_label_dst),
        ]:
            if not os.path.exists(dst):
                os.symlink(src, dst)
                # print(f"已建立符号链接，src{src}，dst{dst}")
            else:
                print(f"跳过已存在的符号链接：{dst}")

    print(f"成功匹配并建立符号链接的文件对数：{len(matches)}")

# 生成 ImageSets
def gen_image_sets(image_dir):
    # 获取所有图像文件（排除子目录）
    image_files = []
    for filename in os.listdir(image_dir):
        full_path = os.path.join(image_dir, filename)
        if os.path.isfile(full_path):
            image_files.append(filename)
    # 提取文件名（不带后缀）
    filenames = [os.path.splitext(f)[0] for f in image_files]
    filenames.sort()
    # 写入四个文件
    for filename in ['train.txt', 'val.txt', 'trainval.txt', 'test.txt']:
        file_path = os.path.join(dst_imagesets_dir, filename)
        with open(file_path, 'w') as f:
            for name in filenames:
                f.write(f"{name}\n")
        print(f"已生成文件：{file_path}")
    
if __name__ == "__main__":
    clean_links(src_dir)
    clean_links(dst_base_dir)
    
    mkdirs(dst_training_dirs)
    mkdirs(dst_testing_dirs)
    matched = find_matchs()
    link_matches(matched)

    os.makedirs(dst_imagesets_dir, exist_ok=True)
    gen_image_sets(dst_training_dirs["image"])
