from pathlib import Path
from PIL import Image

# 定义源目录和目标目录
source_dir = Path('.vscode/raw_blade_data/image')
image_2_dir = Path('.vscode/raw_blade_data/processed/image_2')
label_2_dir = Path('.vscode/raw_blade_data/processed/label_2')

def clear_directory(directory: Path):
    if directory.exists():
        for item in directory.glob('*'):
            if item.is_file():
                item.unlink()  # 删除文件

# 确保目标目录存在
image_2_dir.mkdir(parents=True, exist_ok=True)
clear_directory(image_2_dir)
label_2_dir.mkdir(parents=True, exist_ok=True)
clear_directory(label_2_dir)

# 遍历源目录中的所有 .jpg 文件
for file in source_dir.iterdir():
    if file.is_file() and file.suffix.lower() == '.jpg':
        # 构建目标 PNG 文件路径
        png_file = image_2_dir / (file.stem + '.png')

        # 使用 Pillow 打开并保存为 PNG 格式
        with Image.open(file) as img:
            img.save(png_file, format='PNG')

        # 在 label_2 目录下创建同名的 .txt 文件（去掉后缀）
        txt_file = label_2_dir / (file.stem + '.txt')
        txt_file.touch(exist_ok=True)  # 创建空文件，若已存在则不修改

print("图片转换和空标签文件生成完成。")
