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

  KITTI数据集中easy、moderate、hard根据标注框是否被遮挡、遮挡程度和框的高度进行定义的，具体数据如下：
  
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

**这一段需要重写**，目前还是用Kitti数据集模拟。

```shell
mkdir /root/workspace/mmdetection3d/data/blade/
ln -s /root/workspace/mmdetection3d/.vscode/OpenDataLab___KITTI_Object/raw/testing/calib /root/workspace/mmdetection3d/data/blade/testing/calib
ln -s /root/workspace/mmdetection3d/.vscode/OpenDataLab___KITTI_Object/raw/testing/image_2 /root/workspace/mmdetection3d/data/blade/testing/image_2
ln -s /root/workspace/mmdetection3d/.vscode/OpenDataLab___KITTI_Object/raw/testing/velodyne /root/workspace/mmdetection3d/data/blade/testing/velodyne
ln -s /root/workspace/mmdetection3d/.vscode/OpenDataLab___KITTI_Object/raw/training/calib /root/workspace/mmdetection3d/data/blade/training/calib
ln -s /root/workspace/mmdetection3d/.vscode/OpenDataLab___KITTI_Object/raw/training/image_2 /root/workspace/mmdetection3d/data/blade/training/image_2
ln -s /root/workspace/mmdetection3d/.vscode/OpenDataLab___KITTI_Object/raw/training/velodyne /root/workspace/mmdetection3d/data/blade/training/velodyne
ln -s /root/workspace/mmdetection3d/.vscode/OpenDataLab___KITTI_Object/raw/training/velodyne_reduced /root/workspace/mmdetection3d/data/blade/training/velodyne_reduced
```

处理label类型，操作后只剩下Blade和DontCare类型

```shell
cp -rf /root/workspace/mmdetection3d/.vscode/OpenDataLab___KITTI_Object/raw/training/label_2 /root/workspace/mmdetection3d/data/blade/training/label_2
find data/blade/training/label_2 -type f -name "*.txt" -exec sed -i 's/Car /Blade /g' {} \;
find data/blade/training/label_2 -type f -name "*.txt" -exec sed -i 's/Van/DontCare/g' {} \;
find data/blade/training/label_2 -type f -name "*.txt" -exec sed -i 's/Truck/DontCare/g' {} \;
find data/blade/training/label_2 -type f -name "*.txt" -exec sed -i 's/Pedestrian/DontCare/g' {} \;
find data/blade/training/label_2 -type f -name "*.txt" -exec sed -i 's/Person_sitting/DontCare/g' {} \;
find data/blade/training/label_2 -type f -name "*.txt" -exec sed -i 's/Cyclist/DontCare/g' {} \;
find data/blade/training/label_2 -type f -name "*.txt" -exec sed -i 's/Tram/DontCare/g' {} \;
find data/blade/training/label_2 -type f -name "*.txt" -exec sed -i 's/Misc/DontCare/g' {} \;
```

```shell
python tools/create_data.py blade --root-path ./data/blade --out-dir ./data/blade --extra-tag blade
```

### 3.3. 风电叶片数据集训练

```shell
nohup python tools/train.py configs/pointpillars/pointpillars_wind-turbine-blade.py > train.log 2>&1 &
```

### 3.4. 风电叶片数据集测试

Todo
