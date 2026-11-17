# MDRAMP

[English](README.md) | [简体中文](README-Zh.md)

用于抗菌肽公开数据预训练、迭代校准、MIC 预测与候选序列生成。

## 安装

安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 Conda，在项目根目录执行：

```bash
conda env create -f environment.yml
conda activate mdramp
python main.py --help
```

首次使用会自动从 [MDRAMP](https://huggingface.co/BillyLiang/MDRAMP) 和 [ESM3](https://huggingface.co/biohub/esm3-sm-open-v1) 下载模型。

## 快速开始

```bash
# 预测多肽 MIC
python main.py predict --input demo/predict.fasta

# 使用实验数据校准当前模型
python main.py calibrate --input demo/experiments.csv

# 生成 100 条候选序列
python main.py generate --count 100

# 使用随附公开数据训练
python main.py train
```

结果保存到 `runs/` 下的新目录。通过 `--output-dir DIR` 指定输出位置，使用 `--help` 查看命令选项。

## 子命令说明

所有命令均支持以下公共参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--output-dir DIR` | `runs/` 下自动生成的时间戳目录 | 输出目录，不能已经存在。 |
| `--device {cpu,cuda}` | `cpu` | 计算设备；CUDA 需要 NVIDIA 驱动和支持 CUDA 的 PyTorch。 |
| `--threads N` | `2` | CPU 线程数。 |
| `--seed N` | `11` | 随机种子；公开专家集成使用训练计划中固定的种子列表。 |
| `--help` | — | 查看对应命令的帮助。 |

### 模型训练：`train`

训练公开数据专家和自回归生成器。训练数据由 `data/public/training_plan.json` 指定，ESM3 特征自动计算。例如：

```bash
python main.py train --epochs 100 --batch-size 32 --output-dir runs/public_training
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--epochs N` | `100` | 各训练阶段的轮数上限。 |
| `--batch-size N` | `32` | 训练批大小。 |
| `--learning-rate X` | `0.0001` | 各阶段使用的学习率。 |

**输出：** `experts/`（专家权重与标准化器）、`interaction.pt`、`generator.pt`、`scorer.json`、训练日志和清单。
### 迭代校准：`calibrate`

使用实验观测更新 MIC 预测头，编码器和专家保持冻结。默认在当前注册的 FinalModel 基础上继续训练。

```bash
python main.py calibrate --input demo/experiments.csv --output-dir runs/calibration
# 使用本次校准的模型进行预测
python main.py predict --model runs/calibration/model.pt --input demo/predict.fasta
```

**输入：** CSV或者TSV格式文件，每行一条观测，列名区分大小写。输入文件的每一列的内容和定义如下：

| 列名 | 必需 | 说明 |
|---|---|---|
| `sequence` | 是 | 氨基酸序列。 |
| `mic` | 是 | 原始尺度 MIC，单位 μg/mL，例如 `8` 或 `>128`。 |
| `id` | 否 | 多肽编号。 |
| `strain` | 否 | 实验菌株，用于追溯。 |
| `round` | 否 | 实验轮次。 |

输入示例见 [`demo/experiments.csv`](demo/experiments.csv)。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input FILE` | 必填 | 实验数据表。 |
| `--model MODEL` | `FinalModel` | `FinalModel`、`BaseModel`，或权重/配置文件路径。 |
| `--epochs N` | `100` | 校准轮数。 |
| `--batch-size N` | `32` | 校准批大小。 |
| `--learning-rate X` | `0.0001` | 校准学习率。 |

### 活性预测：`predict`

使用训练的模型，为每条肽输出一个 MIC。

```bash
python main.py predict --input demo/predict.fasta --details --output-dir runs/prediction
```

**输入：** FASTA、CSV、TSV。如果是表格，则必需包括 `sequence` 列。

```csv
id,sequence
Test1,FLRRLVKWLKKM
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input FILE` | 必填 | 序列文件。 |
| `--details` | 关闭 | 额外将专家分数写入 `details.csv`。 |

**输出：** `predictions.csv`，包含 `id`、`sequence`、`predicted_mic`（μg/mL）。输入示例见 [`demo/`](demo/)。

### 候选生成：`generate`

从预训练生成器采样，经筛选、去重并通过 `--exclude FILE` 排除所提供的已测序列后，按所选模型的综合评分 `R_t` 排序。无需输入文件。

```bash
python main.py generate --count 100 --details --output-dir runs/candidates
```

| 参数 | 默认值 | 原理与作用 | 取值范围 | 调整效果 |
|---|---|---|---|---|
| `--model MODEL` | `FinalModel` | 加载模型包中的生成器 | /| /|
| `--count N` | `100` | 从合格候选池中按 `R_t` 降序返回前 N 条。 | 整数，`N ≥ 1`，且不超过 `max-proposals`。 | / |
| `--pool-multiplier N` | `3` | 将目标合格候选池大小设为 `count × N`，再从中选出 `count` 条。 | 整数，`N ≥ 1`；采样受 `max-proposals` 限制。 | 增大可扩大排序选择范围，通常增加计算量|
| `--temperature X` | `1.0` | 每一步将生成器 logits 除以 X，再转换为采样概率。 |`X > 0`。 | 越大，概率分布越平坦，通常更多样、更随机，也可能增加不合格序列；越小，越偏向高概率氨基酸，通常更集中、易重复。`1` 保持原始 logits 尺度。 |
| `--top-p X` | `0.95` | 每一步保留按概率从高到低累计达到 X 的最小词元集合，再从中采样。 | `0 < X ≤ 1`。 | 越大，允许更多低概率选择，通常增加多样性；越小，采样更集中、可能更易重复。|
| `--details` | 关闭 | 将专家分数及综合分数 `R_t` 另存为 `details.csv`。 | /| / |

例如，`--count 100 --pool-multiplier 3` 表示目标是先收集 300 条合格且不重复的序列，再返回得分最高的 100 条。若达到采样上限时只有 100–299 条合格序列，仍返回其中前 100 条；不足 100 条则报错。

温度和 `top-p` 共同控制采样：先用温度调整概率，再执行 `top-p` 截断。

**输出：** `candidates.csv`（`rank`、`id`、`sequence`、`predicted_mic`，MIC 单位 μg/mL）、`candidates.fasta` 和 `run.json`。

使用 NVIDIA GPU 时，先按 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)在 Conda 环境中安装与驱动匹配的 CUDA 版本，再执行：

```bash
python main.py predict --input demo/predict.fasta --device cuda
python main.py calibrate --input demo/experiments.csv --device cuda
python main.py generate --count 100 --device cuda
python main.py train --device cuda
```
