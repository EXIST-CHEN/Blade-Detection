# 训练说明

## 1. 环境准备

### 1.1. 准备开发机

开发机在阿里云上购买，详细参数如下：

| 参数名   | 参数值                              |
| ------- | ----------------------------------- |
| Machine | AlibabaCloud ecs.gn6i-c24g1.6xlarge |
| OS      | Ubuntu 18.04 x64                    |
| Cpu     | 24 Cores                            |
| Memory  | 93 GB                               |
| GPU     | NVIDIA T4 * 1, 16GB                 |
| Cuda    | 10.2                                |

### 1.2. 准备开发环境

1. 下载`mmdetection3d`代码。

    ```shell
    cd /root # 本项目在 root 下开发，但建议不要使用该目录
    mkdir -p workspace # 创建工作目录
    cd workspace
    git clone https://github.com/open-mmlab/mmdetection3d.git # 下载原始代码库
    git fetch --tags
    git checkout v1.1.0rc3 # 切换到 v1.1.0rc3 版本
    cd mmdetection3d
    ```

2. 同步到自己的代码库，并创建开发分支

    ```shell
    git checkout -b blade-det-dev # 将 v1.1.0rc3 版本切出为本地分支
    git remote add github git@github.com:EXIST-CHEN/Blade-Detection.git # 添加个人仓库
    git push -u github blade-det-dev:mmdet3d-v1.1.0rc3 # mmdet3d-v1.1.0rc3作为基础分支
    git push -u github blade-det-dev:blade-det-dev # blade-det-dev作为开发分支
    ```

3. 搭建环境。不能按照官方文档来，有很多版本依赖的坑，下方是实测没问题的。

    ```shell
    conda create --name openmmlab python=3.8 -y # 使用Conda创建隔离的Python虚拟环境，确保依赖包版本一致。如果要删除当前虚拟环境，可以用 `conda env remove -n openmmlab` 命令
    conda activate openmmlab # 进入虚拟环境，每次打开控制台都需要做！！！如果要退出虚拟环境，可以用 `conda deactivate` 命令
    conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch # 安装特定版本的 PyTorch 框架以兼容 CUDA 10.2
    pip install numpy==1.23.5 # 测试发现不能安装最新的 numpy 否则会有依赖问题，必须使用 1.23.5 版本
    pip install -U openmim # 安装 min 用于包管理
    mim install mmengine # 安装 mmengine ，这个目前没有版本的坑，可以直接装
    mim install 'mmcv>=2.0.0rc4,<2.1.0' # 安装 mmcv ， 版本限制比较死，只能在这两个版本间（官方提示的范围稍微大一些）
    mim install 'mmdet>=3.0.0,<3.1.0' # 安装 mmdet ， 版本限制比较死，只能在这两个版本间（官方提示的范围稍微大一些）
    pip install -v -e . # 安装本项目。
    ```

## 2. 验证性训练

验证阶段，使用 KITTI 数据集进行测试，验证开发环境是否正常，以及评估 PointPillars 算法效果。

### 2.1. KITTI 数据集准备

由于 KITTI 数据集在海外 aws s3 上，下载较慢，使用 OpenDataLab 源进行下载。

```shell
cd .vscode
openxlab dataset get --dataset-repo OpenDataLab/KITTI_Object # 下载原始数据到 .vscode 目录
cd OpenDataLab___KITTI_Object/
for zip_file in *.zip; do # 解压数据
    unzip "$zip_file"; 
done
```

将需要的数据链接到 mmdetection3d 的数据目录 `data/kitti/`。创建链接使用 `ln -s <source_file> <dest_file>`命令，删除使用 `rm <dest_file>` 命令。

```shell
cd /root/workspace/mmdetection3d
mkdir /root/workspace/mmdetection3d/data/kitti/
mkdir /root/workspace/mmdetection3d/data/kitti/ImageSets
ln -s /root/workspace/mmdetection3d/.vscode/OpenDataLab___KITTI_Object/raw/testing/calib /root/workspace/mmdetection3d/data/kitti/testing/calib
ln -s /root/workspace/mmdetection3d/.vscode/OpenDataLab___KITTI_Object/raw/testing/image_2 /root/workspace/mmdetection3d/data/kitti/testing/image_2
ln -s /root/workspace/mmdetection3d/.vscode/OpenDataLab___KITTI_Object/raw/testing/velodyne /root/workspace/mmdetection3d/data/kitti/testing/velodyne
ln -s /root/workspace/mmdetection3d/.vscode/OpenDataLab___KITTI_Object/raw/training/calib /root/workspace/mmdetection3d/data/kitti/training/calib
ln -s /root/workspace/mmdetection3d/.vscode/OpenDataLab___KITTI_Object/raw/training/image_2 /root/workspace/mmdetection3d/data/kitti/training/image_2
ln -s /root/workspace/mmdetection3d/.vscode/OpenDataLab___KITTI_Object/raw/training/label_2 /root/workspace/mmdetection3d/data/kitti/training/label_2 
ln -s /root/workspace/mmdetection3d/.vscode/OpenDataLab___KITTI_Object/raw/training/velodyne /root/workspace/mmdetection3d/data/kitti/training/velodyne
ln -s /root/workspace/mmdetection3d/.vscode/OpenDataLab___KITTI_Object/raw/training/velodyne_reduced /root/workspace/mmdetection3d/data/kitti/training/velodyne_reduced
```

此时，`/root/workspace/mmdetection3d` 目录如下：

```shell
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── kitti
│   │   ├── ImageSets
│   │   ├── testing
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── velodyne
│   │   ├── training
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── label_2
│   │   │   ├── velodyne
```

下面开始创建 KITTI 点云数据，首先需要加载原始的点云数据并生成相关的包含目标标签和标注框的数据标注文件，同时还需要为 KITTI 数据集生成每个单独的训练目标的点云数据，并将其存储在 `data/kitti/kitti_gt_database` 的 `.bin` 格式的文件中，此外，需要为训练数据或者验证数据生成 `.pkl` 格式的包含数据信息的文件。

```shell
# 下载待加载的数据列表文件
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt --no-check-certificate --content-disposition -O /root/workspace/mmdetection3d/data/kitti/ImageSets/test.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt --no-check-certificate --content-disposition -O /root/workspace/mmdetection3d/data/kitti/ImageSets/train.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt --no-check-certificate --content-disposition -O /root/workspace/mmdetection3d/data/kitti/ImageSets/val.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate --content-disposition -O /root/workspace/mmdetection3d/data/kitti/ImageSets/trainval.txt

# 生成 .pkl 文件
python /root/workspace/mmdetection3d/tools/create_data.py kitti --root-path /root/workspace/mmdetection3d/data/kitti --out-dir /root/workspace/mmdetection3d/data/kitti --extra-tag kitti
```

处理完成之后的 `/root/workspace/mmdetection3d/data/kitti` 目录如下：

```shell
kitti
├── ImageSets
│   ├── test.txt
│   ├── train.txt
│   ├── trainval.txt
│   ├── val.txt
├── kitti_gt_database
│   ├── xxxxx.bin
├── testing
│   ├── calib
│   ├── image_2
│   ├── velodyne
│   ├── velodyne_reduced
├── training
│   ├── calib
│   ├── image_2
│   ├── label_2
│   ├── velodyne
│   ├── velodyne_reduced
├── kitti_dbinfos_train.pkl
├── kitti_infos_test.pkl
├── kitti_infos_train.pkl
├── kitti_infos_trainval.pkl
├── kitti_infos_val.pkl
```

### 2.2. KITTI 训练

为了精简描述，以下所有操作的默认路径都为 `/root/workspace/mmdetection3d` 目录。

mmdetection3d 中，所有的训练都需要配置一个 config，里面定义了训练相关的参数。这里我们使用已有的 `configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py` 作为测试配置，验证 KITTI 能否跑通。

为了尽可能跑满开发机的 GPU ，需要进行一些训练参数的调整：

* 修改 `configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py` 中的 `epoch_num` 可以调整训练轮数。
* 修改 `configs/_base_/datasets/kitti-3d-3class.py` 中的 `batch_size` 从 6 至 12，`num_workers` 从 4 至 6。
* 修改 `configs/_base_/default_runtime.py` 中的 `load_from` 和 `resume` 参数可以从指定的训练结果继续训练。

按需调整完参数后，执行以下命令，开始训练。

```shell
nohup python tools/train.py configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py > train.log 2>&1 &
```

可以通过以下命令监控资源的使用

```shell
watch -n 1 nvidia-smi # 监控 GPU
htop # 监控 CPU
```

以开发机的性能，跑默认的 80 个 epoch 大约需要 15 小时。训练的实时日志会输出到 `train.log` 中（如果执行失败的话，会包括报错信息），而单次运行的结果会保存在 `work_dirs/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class/` 目录下，里面会有一个 `YYYYMMDD_HHIISS` 目录（执行训练的时间）里面保存了日志，还有一个 `epoch_x.pth` 文件是训练的结果（x为epoch数）。

### 2.3. KITTI 测试与可视化

训练完成后，可以执行测试命令。

```shell
nohup python tools/test.py configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py work_dirs/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class/epoch_50.pth 2>&1 & # 结果在 work_dirs/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class/20250427_104443-test
```

得到的结果如下：

```text
----------- AP11 Results ------------

Pedestrian AP11@0.50, 0.50, 0.50:
bbox AP11:65.3059, 62.0850, 59.0062
bev  AP11:58.7979, 53.4739, 49.2157
3d   AP11:53.7615, 48.3636, 44.1112
aos  AP11:43.40, 41.22, 38.96
Pedestrian AP11@0.50, 0.25, 0.25:
bbox AP11:65.3059, 62.0850, 59.0062
bev  AP11:72.8516, 69.4336, 65.0426
3d   AP11:72.4786, 68.4898, 64.4587
aos  AP11:43.40, 41.22, 38.96
Cyclist AP11@0.50, 0.50, 0.50:
bbox AP11:84.1006, 72.8933, 71.1513
bev  AP11:82.0529, 66.0701, 62.3619
3d   AP11:79.0043, 61.5962, 57.0006
aos  AP11:83.26, 71.02, 69.00
Cyclist AP11@0.50, 0.25, 0.25:
bbox AP11:84.1006, 72.8933, 71.1513
bev  AP11:84.1775, 71.7558, 69.4283
3d   AP11:84.1775, 71.7553, 69.3786
aos  AP11:83.26, 71.02, 69.00
Car AP11@0.70, 0.70, 0.70:
bbox AP11:90.7415, 89.4125, 86.4478
bev  AP11:89.5794, 86.6012, 79.4082
3d   AP11:86.1992, 76.6481, 73.6870
aos  AP11:90.67, 89.02, 85.72
Car AP11@0.70, 0.50, 0.50:
bbox AP11:90.7415, 89.4125, 86.4478
bev  AP11:90.7541, 89.9228, 89.1394
3d   AP11:90.7541, 89.8350, 88.8213
aos  AP11:90.67, 89.02, 85.72

Overall AP11@easy, moderate, hard:
bbox AP11:80.0494, 74.7969, 72.2018
bev  AP11:76.8101, 68.7151, 63.6619
3d   AP11:72.9883, 62.2026, 58.2662
aos  AP11:72.44, 67.09, 64.56

----------- AP40 Results ------------

Pedestrian AP40@0.50, 0.50, 0.50:
bbox AP40:65.9888, 62.1628, 58.1800
bev  AP40:58.4884, 52.5665, 48.0129
3d   AP40:53.1469, 46.5298, 41.9914
aos  AP40:44.08, 41.54, 38.70
Pedestrian AP40@0.50, 0.25, 0.25:
bbox AP40:65.9888, 62.1628, 58.1800
bev  AP40:73.4284, 70.1504, 65.6691
3d   AP40:73.0839, 69.0382, 64.9700
aos  AP40:44.08, 41.54, 38.70
Cyclist AP40@0.50, 0.50, 0.50:
bbox AP40:86.0735, 74.2598, 71.2538
bev  AP40:84.4753, 66.1594, 62.2377
3d   AP40:80.1116, 60.7656, 56.9344
aos  AP40:85.15, 72.14, 68.87
Cyclist AP40@0.50, 0.25, 0.25:
bbox AP40:86.0735, 74.2598, 71.2538
bev  AP40:87.1643, 72.8538, 69.6298
3d   AP40:87.1640, 72.8517, 69.5880
aos  AP40:85.15, 72.14, 68.87
Car AP40@0.70, 0.70, 0.70:
bbox AP40:95.8660, 92.2436, 87.4879
bev  AP40:92.2258, 88.0301, 83.4938
3d   AP40:87.8618, 76.6546, 73.5539
aos  AP40:95.77, 91.81, 86.73
Car AP40@0.70, 0.50, 0.50:
bbox AP40:95.8660, 92.2436, 87.4879
bev  AP40:96.0451, 94.9365, 90.2321
3d   AP40:96.0132, 94.6836, 90.0482
aos  AP40:95.77, 91.81, 86.73

Overall AP40@easy, moderate, hard:
bbox AP40:82.6428, 76.2221, 72.3073
bev  AP40:78.3965, 68.9187, 64.5815
3d   AP40:73.7067, 61.3167, 57.4932
aos  AP40:75.00, 68.50, 64.77
```

对于上述的结果，下面分别对特定的名词进行解释说明：

* bbox、bev、3d、aos
  
  深度学习算法的检测指标通常由 bbox、bev、3d、aos四个检测指标，其含义分别如下所示：
  * bbox：2D 检测框的准确率
  * bev：BEV 视图下检测框的准确率
  * 3d：3D 检测框的准确率
  * aos：检测目标旋转角度的准确率

* AP11 与 AP40：

  * AP11：表示 11 点插值平均精度，在 KITTI 3D中 `R11={0,0.1,0.2，……，1}`，是等间距的 recall level
  * AP40：表示 40 点插值平均精度，将 R11 修改为 `R40={1/40，2/40，3/40，……，1}`，同样是等间距的 recall level

  论文《Disentangling Monocular 3D Object Detection》证明 AP11 是不准确的，因为当模型可以提供一个精度极小，仅仅是>0的一个单一目标，此时R=0时的精度即为1，那么AP11的平均精度即为1/11，这个精度已经超过了很多的方法，所以是不合理的。所以后续修改为AP40。

* `Car AP11@0.70, 0.70, 0.70` 与 `Car AP11@0.70, 0.50, 0.50`

  * `AP11@0.70, 0.70, 0.70`分别代表 bbox，bev，3d 在 0.70 阈值下的平均精度
  * `AP11@0.70, 0.50, 0.50`分别代表 bbox，bev，3d 在 0.70,0.50,0.50 不同阈值下的平均精度

  这里可以发现，评估 bbox 只有0.70这个阈值，所以可以发现 bbox 的两行数据都是一样的，而对于 bev 与 3d 来说，0.50 的阈值比 0.70 的阈值要宽松，所以第二组的结果（阈值0.50）一般要比第一组的结果（阈值0.70）要高。

* `bbox AP11:xxxx, yyyy, zzzz`

  无论是 bbox，还是 bev、3d、aos，每个评价指标在某一个阈值下都会有 3 组结果，这三组结果分别对应的是 easy、moderate 和 hard 下的评估结果。难度越来越大，所以数值也越来越小，所以这三组数值一般是呈递减状态。

* mAP

  一般论文中的实验结果都会贴上一个 mAP 的最终结果，这个结果就是 moderate mAP 的结果。比如，在刚刚的 PointPillars 实验结果中，对于 car 类别的 AP11 结果如下所示，其中 76.6481 就是作为 car 这个类基准排名的主要指标。

  ```text
  Car AP11@0.70, 0.70, 0.70:
  3d   AP11:86.1992, 76.6481, 73.6870
  ```

  而对于全部 3 个类别的 AP11 结果如下所示，那么 62.2026 就是作为 3 类（3 Class）基准排名的主要指标。

  ```text
  Overall AP11@easy, moderate, hard:
  3d   AP11:72.9883, 62.2026, 58.2662
  ```

  mmdetection3d 中 PointPillars 的结果如下：

  ![alt text](https://private-user-images.githubusercontent.com/210235822/440433877-99e6fb7d-6804-4e47-9223-3ca7360c631a.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDY0NTkyNjYsIm5iZiI6MTc0NjQ1ODk2NiwicGF0aCI6Ii8yMTAyMzU4MjIvNDQwNDMzODc3LTk5ZTZmYjdkLTY4MDQtNGU0Ny05MjIzLTNjYTczNjBjNjMxYS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNTA1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDUwNVQxNTI5MjZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1hZWJmODE0MzJhYjQ5MGIwNGE1Mzk2MDg5NzdkMThlNzUxNmZkN2UzOTYzY2IwNzk3ZDcxOTZmZDQzMWUyNjQ2JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.sl1oBNe5HDm26d6AR7mj6u0l_WJ3JEzQCzJOz1Wligo)

  可以发现，刚刚运行得出的结果 Class 的 AP 是 76.6481，而 3 Class 的 AP 是 62.2026。这个结果与官方跑出来的 77.6 和64.07 接近，所以训练期间的验证结果还是可信的。

* 结果中 easy、moderate、hard 的定义

  KITTI 数据集中 easy、moderate、hard 根据标注框是否被遮挡、遮挡程度和框的高度进行定义的，具体数据如下：
  
  * 简单：最小边界框高度：40 像素，最大遮挡级别：完全可见，最大截断：15%
  * 中等：最小边界框高度：25 像素，最大遮挡水平：部分遮挡，最大截断：30%
  * 困难：最小边界框高度：25 像素，最大遮挡级别：难以看到，最大截断：50%

## 3. 实际训练

所有代码变更见 [PR](https://github.com/EXIST-CHEN/Blade-Detection/pull/1)，下文只列举需要做什么，不会详细列出代码。

### 3.1. 准备风电叶片数据集适配代码

将风电叶片数据集命名为 Blade。

1. 添加数据集定义

    添加文件 `mmdet3d/datasets/blade_dataset.py`，作为该数据集的定义。

    ```python
    # 需要将 BladeDataset 注册到 mmdet3d 库里
    @DATASETS.register_module()
    class BladeDataset(Det3DDataset):
        METAINFO = {
            'classes': ('Blade', ) # 这里，我们只需要一种类型，即风电叶片
        }
        # 其他初始化操作
        ...
    ```

    然后，将该数据集添加到 `mmdet3d/datasets/__init__.py` 中，并在 `mmdet3d/datasets/convert_utils.py` 中添加对应的处理方法。

2. 添加数据集预处理逻辑

    数据集的预处理是通过 `tools/create_data.py` 进行的，这里需要添加 Blade 数据集的预处理逻辑。

    预处理逻辑的相关代码保存在：

    * `tools/dataset_converters/blade_converter.py`
    * `tools/dataset_converters/blade_data_utils.py`
    * `tools/dataset_converters/update_infos_to_v2.py`

    同时，也需要在 `tools/dataset_converters/create_gt_database.py` 中添加对 Blade 数据集的支持。

3. 添加数据集评估逻辑

    配套的评估函数等，都统一写在了 `mmdet3d/evaluation/functional/blade_utils` 目录下，并注册到框架中（`mmdet3d/evaluation/__init__.py`）。

    需要添加 `BladeMetric` 类（`mmdet3d/evaluation/metrics/blade_metric.py`），并将其注册到框架中（`mmdet3d/evaluation/metrics/__init__.py`）。

4. 添加数据集训练配置

    mmdetection3d 中，训练、测试、评估都需要有一个配置，为 Blade 模型也创建了一个配置 `configs/pointpillars/pointpillars_wind-turbine-blade.py`，里面主要定义了数据目录、数据集类型、模型参数、训练流水线、测试流水线、评估流水线等参数。

    ```python
    dataset_type = 'BladeDataset'
    data_root = 'data/blade/'
    class_names = ['Blade']
    lr = 0.001
    epoch_num = 80
    ...
    model = dict(...)
    db_sampler = dict(...)
    train_pipeline = [...]
    test_pipeline = [...]
    eval_pipeline = [...]
    train_dataloader = dict(...)
    val_dataloader = dict(...)
    test_dataloader = dict(...)
    val_evaluator = dict(...)
    test_evaluator = dict(...)
    vis_backends = [...]
    visualizer = dict(...)
    optim_wrapper = dict(...)
    param_scheduler = [...]
    train_cfg = dict(...)
    val_cfg = dict()
    test_cfg = dict()
    auto_scale_lr = dict(...)
    ```

    至此，风电叶片数据集 Blade 就可以在 mmdetection3d 运行了。在运行之前，需要先准备数据集。

### 3.2. 准备风电叶片数据集

将原始数据保存到 `.vscode/raw_blade_data`，目录下的文件为：

```text
mmdetection3d
├── .vscode
│   ├── raw_blade_data
│   │   ├── image
│   │   │   ├── xxxxxx.xxxxx.jpg
│   │   ├── pointcloud
│   │   │   ├── 2001
│   │   │   │   ├── xxxxxx.xxxxx.pcd
│   │   │   ├── 4001
│   │   │   │   ├── xxxxxx.xxxxx.pcd
│   │   │   ├── 5001
│   │   │   │   ├── xxxxxx.xxxxx.pcd
│   │   │   ├── rslidar
│   │   │   │   ├── xxxxxx.xxxxx.pcd
```

需要对这些数据进行处理，使其满足 KITTI 数据集格式，便于被 mmdetection3d 框架加载并进行训练。

#### 3.2.1. 对图像数据进行打标处理

目标：生成 `.vscode/raw_blade_data/processed/image_2` 和 `.vscode/raw_blade_data/processed/label_2` 目录。

处理逻辑：见 `blade_data_process/process_images.py`，需要将 jpg 转换为 png 格式。

```python
from pathlib import Path
from PIL import Image

# 定义源目录和目标目录
source_dir = Path('.vscode/raw_blade_data/image')
image_2_dir = Path('.vscode/raw_blade_data/training/image_2')
label_2_dir = Path('.vscode/raw_blade_data/training/label_2')

# 确保目标目录存在
image_2_dir.mkdir(parents=True, exist_ok=True)
label_2_dir.mkdir(parents=True, exist_ok=True)

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

```

对 `.vscode/raw_blade_data/processed/label_2` 目录下的文件进行手动标注。label 每行包含 15 个字段，用空格隔开，依次的含义如下：

| 字段   | 名称                                         | 类型           | 说明                                                                 |
| ----- | -------------------------------------------- | ------------- | ------------------------------------------------------------------- |
| 1     | type                                         | string        | 物体类别，本场景只有 Blade                                             |
| 2     | truncated                                    | float [0,1]   | 截断程度，表示目标是否部分超出图像边界（0 表示完整可见），本场景均为 0          |
| 3     | occluded                                     | int [0,4]     | 遮挡程度，0=未遮挡，1=部分遮挡，2=大部分遮挡，3=完全遮挡，4=未知，本场景均为 0 |
| 4     | alpha                                        | float [-π, π] | 观测角度（rad），即目标中心与相机光轴之间的角度（用于估计方向）                |
| 5-8   | bbox_left, bbox_top, bbox_right, bbox_bottom | float         | 2D 边界框坐标（像素），表示目标在图像中的矩形区域                            |
| 9-11  | h, w, l                                      | float         | 3D 尺寸（米），分别表示物体的高度（height）、宽度（width）、长度（length）      |
| 12-14 | x, y, z                                      | float         | 3D 位置（米），表示物体在相机坐标系下的中心坐标（x 向右，y 向下，z 向前）       |
| 15    | ry                                           | float [-π, π] | 绕 y 轴旋转角度（rad），表示物体的朝向（yaw）                              |

手动标注前，可以预写入模版。

```shell
python blade_data_process/write_template.py .vscode/raw_blade_data/processed/label_2 "Blade 0.00 0 0.52 0.00 <bbox_top> 640.00 <bbox_bottom> 30.00 0.10 2.00 <x> <y> <z> <ry>
" -y
```

#### 3.2.2. 对雷达数据进行打标处理

目标：生成 `.vscode/raw_blade_data/processed/velodyne_reduced` 和 `.vscode/raw_blade_data/processed/calib` 目录。

处理逻辑：见 `blade_data_process/process_lidar.py`。

```python
from pathlib import Path
import open3d as o3d
import numpy as np

# 定义路径
source_dir = Path('.vscode/raw_blade_data/pointcloud/rslidar')
velodyne_dir = Path('.vscode/raw_blade_data/training/velodyne')
velodyne_reduced_dir = Path('.vscode/raw_blade_data/training/velodyne_reduced')
calib_dir = Path('.vscode/raw_blade_data/training/calib')

# 创建目标目录
velodyne_dir.mkdir(parents=True, exist_ok=True)
velodyne_reduced_dir.mkdir(parents=True, exist_ok=True)
calib_dir.mkdir(parents=True, exist_ok=True)

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

        # 原始点云保存为 BIN（包含所有字段）
        points = np.asarray(pcd.points)
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            points = np.hstack((points, colors))
        elif pcd.has_normals():
            normals = np.asarray(pcd.normals)
            points = np.hstack((points, normals))

        # 保存原始点云到 velodyne 目录
        bin_file_velodyne = velodyne_dir / (pcd_file.stem + '.bin')
        points.astype(np.float32).tofile(str(bin_file_velodyne))
        print(f"原始点数：{points.shape[0]}")

        # 降采样处理
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)

        # 检查降采样后是否为空
        if len(pcd_downsampled.points) == 0:
            print(f"警告：{pcd_file.name} 降采样后无有效点，跳过保存。")
            continue

        # 仅保留 xyz 字段
        points_reduced = np.asarray(pcd_downsampled.points)

        # 保存降采样后的点云到 velodyne_reduced 目录（仅 xyz）
        bin_file_reduced = velodyne_reduced_dir / (pcd_file.stem + '.bin')
        points_reduced.astype(np.float32).tofile(str(bin_file_reduced))
        print(f"降采样后点数：{points_reduced.shape[0]}")

        # 生成空的 calib 文件
        txt_file = calib_dir / (pcd_file.stem + '.txt')
        txt_file.touch(exist_ok=True)
```

对 `.vscode/raw_blade_data/processed/calib` 目录下的文件进行手动标注。calib 的数据格式如下：

* 相机内参矩阵（P0 - P3）：每个相机对应一个 3x4 的投影矩阵，用于将 3D 点（相机坐标系）投影到 2D 图像坐标系。
* 矫正旋转矩阵（R0_rect）：用于将点从原始相机坐标系转换到矫正后的相机坐标系（去除镜头畸变）。
* LiDAR 到相机的坐标变换（Tr_velo_to_cam）：将点从激光雷达坐标系（Velodyne）转换到相机坐标系。
* IMU 到 LiDAR 的坐标变换（Tr_imu_to_velo）：将点从 IMU 坐标系转换到激光雷达坐标系。

手动标注前，可以预写入模版。

```shell
python blade_data_process/write_template.py .vscode/raw_blade_data/processed/calib "P0: 1.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P1: 1.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P2: 505.603861 0.000000 312.618312 0.000000  505.845117 239.323011 1.805066000000e+02 -3.454157000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 4.981016000000e-03
P3: 1.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
R0_rect: 1.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00
Tr_velo_to_cam:  -1.397395e-01 1.587947e-01 9.773725e-01 1.057611e+00 -8.245766e-01 -5.651492e-01 -2.607322e-02 -1.016234e+00 5.482210e-01 -8.095620e-01 2.099121e-01 -2.244096e-01
Tr_imu_to_velo: 1.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
" -y
```

#### 3.2.3. 预处理数据集

由于雷达帧数小于相机，所以雷达数据量小于图片数据量，需要以雷达为基准，找出时间最接近的照片，并统一重命名，保存到 `data/blade/training` 目录下，然后生成 `.vscode/raw_blade_data/testing` 和 `.vscode/raw_blade_data/ImageSets` 目录，并执行脚本进行预处理：

```shell
python /root/workspace/mmdetection3d/blade_data_process/gen_dataset.py
```

处理后的label结果如下：

```shell
lrwxrwxrwx 1 root root   95 May  6 21:14 000000.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657003.487303495.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000001.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657003.588418484.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000002.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657003.688452244.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000003.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657003.788029671.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000004.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657003.887953520.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000005.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657003.988467455.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000006.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657004.089075327.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000007.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657004.187564611.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000008.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657004.287853003.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000009.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657004.387794256.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000010.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657004.489051104.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000011.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657004.589433193.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000012.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657004.688311577.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000013.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657004.787743807.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000014.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657004.883780479.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000015.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657004.983843565.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000016.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657005.083053589.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000017.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657005.184194803.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000018.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657005.284075260.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000019.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657005.384202003.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000020.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657005.484170437.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000021.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657005.584133625.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000022.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657005.684117079.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000023.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657005.784183741.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000024.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657005.884112835.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000025.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657005.984099627.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000026.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657006.083811998.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000027.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657006.183945656.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000028.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657006.283957005.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000029.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657006.384624004.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000030.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657006.483367920.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000031.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657006.584029198.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000032.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657006.684156895.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000033.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657006.784188509.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000034.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657006.884016275.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000035.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657006.984391689.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000036.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657007.084037304.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000037.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657007.184182644.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000038.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657007.283647776.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000039.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657007.383998156.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000040.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657007.484052181.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000041.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657007.583975792.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000042.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657007.683998346.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000043.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657007.784032106.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000044.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657007.883947134.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000045.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657007.984000683.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000046.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657008.083625793.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000047.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657008.184366226.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000048.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657008.284256935.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000049.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657008.383749962.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000050.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657008.483786106.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000051.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657008.583900690.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000052.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657008.683707237.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000053.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657008.783722401.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000054.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657008.883769274.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000055.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657008.984468937.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000056.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657009.083858013.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000057.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657009.183685303.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000058.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657009.283723116.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000059.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657009.382873297.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000060.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657009.484683037.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000061.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657009.584045172.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000062.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657009.684025288.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000063.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657009.783355951.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000064.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657009.884117603.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000065.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657009.984089136.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000066.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657010.084163427.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000067.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657010.184420586.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000068.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657010.284574986.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000069.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657010.383899212.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000070.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657010.483991861.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000071.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657010.583942175.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000072.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657010.683344126.txt
lrwxrwxrwx 1 root root   95 May  6 21:14 000073.txt -> /root/workspace/mmdetection3d/.vscode/raw_blade_data/processed/label_2/1731657010.784007072.txt
```

预处理数据：

```shell
python tools/create_data.py blade --root-path ./data/blade --out-dir ./data/blade --extra-tag blade
```

处理完成之后的 `/root/workspace/mmdetection3d/data/blade` 目录如下：

```text
blade
├── blade_gt_database
│   ├── xxxxx.bin
├── ImageSets
│   ├── test.txt
│   ├── train.txt
│   ├── trainval.txt
│   ├── val.txt
├── testing
│   ├── calib
│   ├── image_2
│   ├── velodyne_reduced
├── training
│   ├── calib
│   ├── image_2
│   ├── label_2
│   ├── velodyne_reduced
├── blade_dbinfos_train.pkl
├── blade_infos_test.pkl
├── blade_infos_train.pkl
├── blade_infos_trainval.pkl
├── blade_infos_val.pkl
```

### 3.3. 风电叶片数据集训练

使用 5.4.1 节改造后的 MMDetection3D 框架，对预处理后的 Blade 数据集进行训练：

```shell
nohup python tools/train.py configs/pointpillars/pointpillars_wind-turbine-blade.py > train.log 2>&1 &
```

由于使用KITTI数据集测试时，估算出了开发机的性能上限，根据预估值调整了训练的 batch_size 和 num_workers， 故训练过程可以看到，GPU的性能利用率达到了接近100%，可以更高效地训练模型。

![alt text](05d5d81871bd7733c316e1ef50dd8a3.png)

在训练过程中，MMDetection3D 框架会给出预计的训练完成时间和当前的梯度、损失值等。下图是训练过程的截图，可以看到当前运行的是第9个epoch，预计剩余时间是12小时40分钟左右，梯度已经较低，loss也收敛到较低水平。

![alt text](image.png)

Blade 数据集在经过了80个epoch，大约15小时的训练后，得到了较好的性能。

### 3.4. 风电叶片数据集测试

结果如下：

```text
----------- AP11 Results ------------

Blade AP11@0.70, 0.70, 0.70:
bbox AP11:90.7776, 89.4898, 86.1475
bev  AP11:88.7656, 85.7175, 78.7043
3d   AP11:85.7989, 76.1519, 72.8707
aos  AP11:90.71, 89.16, 85.58
Blade AP11@0.70, 0.50, 0.50:
bbox AP11:90.7776, 89.4898, 86.1475
bev  AP11:89.9512, 88.9486, 88.1642
3d   AP11:89.9512, 88.8362, 87.9556
aos  AP11:90.71, 89.16, 85.58

----------- AP40 Results ------------

Blade AP40@0.70, 0.70, 0.70:
bbox AP40:95.8777, 92.2285, 87.5594
bev  AP40:91.0857, 86.8972, 82.4981
3d   AP40:86.8003, 77.3468, 73.0951
aos  AP40:95.78, 91.86, 86.97
Blade AP40@0.70, 0.50, 0.50:
bbox AP40:95.8777, 92.2285, 87.5594
bev  AP40:94.7470, 93.4893, 90.6973
3d   AP40:94.7412, 93.1625, 88.8195
aos  AP40:95.78, 91.86, 86.97
```
