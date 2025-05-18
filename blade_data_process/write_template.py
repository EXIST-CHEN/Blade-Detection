import argparse
from pathlib import Path

# 批量标注图片数据集模板
#
# python blade_data_process/write_template.py .vscode/raw_blade_data/processed/label_2 "Blade 0.00 0 0.52 0.00 <bbox_top> 640.00 <bbox_bottom> 30.00 0.10 2.00 <x> <y> <z> <ry>
# " -y

# 批量标注雷达数据集模板
#
# python blade_data_process/write_template.py .vscode/raw_blade_data/processed/calib "P0: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 0.000000000000e+00 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
# P1: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 -3.797842000000e+02 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
# P2: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 4.575831000000e+01 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 -3.454157000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 4.981016000000e-03
# P3: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 -3.341081000000e+02 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 2.330660000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 3.201153000000e-03
# R0_rect: 9.999128000000e-01 1.009263000000e-02 -8.511932000000e-03 -1.012729000000e-02 9.999406000000e-01 -4.037671000000e-03 8.470675000000e-03 4.123522000000e-03 9.999556000000e-01
# Tr_velo_to_cam: 6.927964000000e-03 -9.999722000000e-01 -2.757829000000e-03 -2.457729000000e-02 -1.162982000000e-03 2.749836000000e-03 -9.999955000000e-01 -6.127237000000e-02 9.999753000000e-01 6.931141000000e-03 -1.143899000000e-03 -3.321029000000e-01
# Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01
# " -y

def write_to_txt_files(directory, content, recursive=False, dry_run=False, confirm=True):
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"错误：目录 {directory} 不存在。")
        return

    # 确认执行
    if confirm:
        answer = input(f"即将在目录 {directory} 下覆盖所有 .txt 文件的内容为：\n{content}\n是否继续？(y/n): ").strip().lower()
        if answer != 'y':
            print("操作已取消。")
            return

    # 收集所有 .txt 文件
    if recursive:
        files = dir_path.rglob("*.txt")
    else:
        files = dir_path.glob("*.txt")

    success_count = 0
    fail_count = 0

    for file in files:
        try:
            if not dry_run:
                with open(file, 'w', encoding='utf-8') as f:
                    f.write(content)
            print(f"{'[模拟] ' if dry_run else ''}已处理文件：{file}")
            success_count += 1
        except Exception as e:
            print(f"写入失败：{file}，错误：{e}")
            fail_count += 1

    print(f"\n✅ 成功处理 {success_count} 个文件")
    if fail_count > 0:
        print(f"❌ 写入失败 {fail_count} 个文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量覆盖写入指定目录下的所有 .txt 文件")
    parser.add_argument("directory", help="目标目录路径")
    parser.add_argument("content", help="要写入的内容")
    parser.add_argument("--recursive", "-r", action="store_true", help="递归处理子目录中的 .txt 文件")
    parser.add_argument("--dry-run", "-d", action="store_true", help="仅列出匹配的文件，不实际写入")
    parser.add_argument("--no-confirm", "-y", action="store_true", help="跳过确认步骤，直接执行")

    args = parser.parse_args()

    write_to_txt_files(
        directory=args.directory,
        content=args.content,
        recursive=args.recursive,
        dry_run=args.dry_run,
        confirm=not args.no_confirm
    )
