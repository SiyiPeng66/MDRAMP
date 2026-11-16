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

运行环境：Python 3.12，CPU。

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

## 模型

| 模型 | 用途 |
|---|---|
| **BaseModel** | 公开数据训练的专家与生成器；需经实验校准后才能预测 MIC。 |
| **FinalModel** | 校准后的模型，默认用于预测、继续校准及候选评分。 |

通过 `--model BaseModel` 或 `--model path/to/model.pt` 指定其他起始模型。自定义权重需附带 `scorer.json`。校准默认保存新模型，增加 `--update-final` 才会将其设为默认模型。

## 子命令说明

所有命令均支持以下公共参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--output-dir DIR` | `runs/` 下自动生成的时间戳目录 | 输出目录，不能已经存在。 |
| `--threads N` | `2` | CPU 线程数。 |
| `--seed N` | `11` | 随机种子；公开专家集成使用训练计划中固定的种子列表。 |
| `--help` | — | 查看对应命令的帮助。 |

### 模型训练：`train`

训练公开数据专家和自回归生成器。训练数据由 `data/public/training_plan.json` 指定，ESM3 特征自动计算。

```bash
python main.py train --epochs 100 --batch-size 32 --output-dir runs/public_training
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--epochs N` | `100` | 各训练阶段的轮数上限。 |
| `--batch-size N` | `32` | 训练批大小。 |
| `--learning-rate X` | `0.0001` | 各阶段使用的学习率。 |

**输出：** `experts/`（专家权重与标准化器）、`interaction.pt`、`generator.pt`、`scorer.json`、训练日志和清单。新模型没有 MIC 预测头，不会替换默认模型。可用以下命令初始化其 MIC 预测头：

```bash
python main.py calibrate --model runs/public_training/scorer.json --input demo/experiments.csv
```

### 迭代校准：`calibrate`

使用实验观测更新 MIC 预测头，编码器和专家保持冻结。默认在当前注册的 FinalModel 基础上继续训练。

```bash
python main.py calibrate --input demo/experiments.csv --output-dir runs/calibration
# 使用本次校准的模型进行预测
python main.py predict --model runs/calibration/model.pt --input demo/predict.fasta
```

**输入：** CSV、TSV 或 XLSX（首个工作表），每行一条观测，列名区分大小写。

| 列名 | 必需 | 说明 |
|---|---|---|
| `sequence` | 是 | 氨基酸序列。 |
| `mic` | 是 | 原始尺度 MIC，单位 μg/mL，例如 `8` 或 `>128`。 |
| `id` | 否 | 多肽编号。 |
| `strain` | 否 | 实验菌株，用于追溯。 |
| `round` | 否 | 实验轮次。 |

模板见 [`templates/experiments.csv`](templates/experiments.csv)，示例见 [`demo/experiments.csv`](demo/experiments.csv)。每条序列使用跨菌株最优 MIC 区间，默认标签**除以 10**，请勿提前缩放输入。使用 `--mic-divisor 1` 可保留原始尺度。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input FILE` | 必填 | 实验数据表。 |
| `--model MODEL` | `FinalModel` | `FinalModel`、`BaseModel`，或权重/配置文件路径。 |
| `--epochs N` | `100` | 校准轮数。 |
| `--batch-size N` | `32` | 校准批大小。 |
| `--learning-rate X` | `0.0001` | 校准学习率。 |
| `--mic-divisor X` | `10` | 训练 MIC 标签使用的正数除数。 |
| `--update-final` | 关闭 | 校准成功后，将新模型注册为默认 FinalModel。 |

**输出：** `model.pt`、`scorer.json`、`calibration.csv`（`id`、`sequence`、`before_mic`、`after_mic`）、处理后的观测/标签及训练追溯记录。从 BaseModel 初始化时，`before_mic` 为空。移动校准模型时，应保留其配置引用的共享资产。

### 活性预测：`predict`

使用 FinalModel 或自定义校准模型，为每条肽输出一个 MIC。BaseModel 必须先校准。

```bash
python main.py predict --input demo/predict.fasta --details --output-dir runs/prediction
```

**输入：** FASTA、CSV、TSV 或 XLSX（首个工作表）。表格必需 `sequence` 列，可选 `id`；FASTA 标题作为编号。仅接受 20 种标准氨基酸，自动去空白并转大写，不需要菌株信息。

```csv
id,sequence
Test1,FLRRLVKWLKKM
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input FILE` | 必填 | 序列文件。 |
| `--model MODEL` | `FinalModel` | 注册模型名或校准权重/配置路径。 |
| `--details` | 关闭 | 额外将专家分数写入 `details.csv`。 |

**输出：** `predictions.csv`，包含 `id`、`sequence`、`predicted_mic`（μg/mL）；`run.json` 记录参数和资产哈希。输入示例及参考预测见 [`demo/`](demo/README.md)。

### 候选生成：`generate`

从预训练生成器采样，经筛选、去重并通过 `--exclude FILE` 排除所提供的已测序列后，按所选模型的综合评分 `R_t` 排序。无需输入文件。

```bash
python main.py generate --count 100 --details --output-dir runs/candidates
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model MODEL` | `FinalModel` | 候选评分模型，必须具备 MIC 预测头。 |
| `--count N` | `100` | 返回的候选数量。 |
| `--pool-multiplier N` | `3` | 合格候选池目标相对 `count` 的倍数，批量采样可能超过目标。 |
| `--max-proposals N` | `20000` | 筛选前最多采样的序列数。 |
| `--temperature X` | `1.0` | 采样温度，必须大于 0。 |
| `--top-p X` | `0.95` | 核采样累计概率，`0 < X ≤ 1`。 |
| `--details` | 关闭 | 额外将专家分数和 `R_t` 写入 `details.csv`。 |

**输出：** `candidates.csv`（`rank`、`id`、`sequence`、`predicted_mic`，MIC 单位 μg/mL）、`candidates.fasta` 和 `run.json`。合格候选不足时，可增加 `--max-proposals`。

## 模型与数据说明

FinalModel 使用调整后的 MIC 标签及排名偏好拟合，输出属于模型估计，不是新增实验测量。候选生成使用预训练自回归生成器采样、过滤，再由所选模型排序，不会在每次请求时重新训练生成器。Demo 序列参与过校准，不是独立测试集。

资产版本、训练输入、候选表格和复现细节见[模型与数据说明](docs/MODELS_AND_DATA-Zh.md)。上游模型与数据集保留各自许可。

## 测试

```bash
python -m pytest -q
```

## 研究数据

完整实验数据及其候选预测表随论文补充材料提供，不纳入本仓库。仓库仅保留已选定的六条 Demo 序列用于测试。生成时可用 `--exclude FILE` 指定 FASTA 或表格文件，排除其中的已测序列。
