import os
import math

# 输入文件路径
input_file = '/root/workspace/mmdetection3d/.vscode/shoudong'
# 输出目录路径
output_dir = '/root/workspace/mmdetection3d/data/blade/training/label_2'

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 读取输入文件
with open(input_file, 'r') as f:
    lines = f.readlines()

# 遍历每一行数据
for idx, line in enumerate(lines):
    parts = line.strip().split()
    if len(parts) != 6:
        print(f"Invalid line {idx}: {line}")
        continue
    
    # 解析输入数据
    bbox_top = int(parts[0])
    second_num = int(parts[1])
    third_num = int(parts[2])
    bbox_bottom = int(parts[3])
    x_pixel = int(parts[4])
    z_pixel = int(parts[5])
    
    # 转换像素到米
    x_m = "{:.2f}".format(x_pixel / 24)
    z_m = "{:.2f}".format(z_pixel / 24)
    
    
    # 构建输出文件名（六位编号）
    filename = f"{idx:06d}.txt"
    output_path = os.path.join(output_dir, filename)
    
    # 填充模板并写入文件
    template = (
        "Blade 0.00 0 0.52 0.00 {} 640.00 {} 30.00 0.10 2.00 {} 10.00 {} 0.00"
    )
    filled_line = template.format(
        "{:.2f}".format(bbox_top),
        "{:.2f}".format(bbox_bottom),
        x_m,
        z_m
    )

    print(filled_line)
    
    with open(output_path, 'w') as out_f:
        out_f.write(filled_line)

print("Processing complete.")
