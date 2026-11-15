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

默认使用 CPU 和 Python 3.12。已在 macOS 验证；代码支持 Linux，但尚未独立实机验证。Windows 请使用 WSL2。

首次使用会自动从 [MDRAMP](https://huggingface.co/BillyLiang/MDRAMP) 和 [ESM3](https://huggingface.co/biohub/esm3-sm-open-v1) 下载模型。全部资产约占 5.75 GiB，后续复用本地文件，文件齐全后可离线运行。

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

结果保存到 `runs/` 下的新目录。通过 `--output-dir DIR` 指定输出位置，使用 `--help` 查看命令选项。公开数据训练会自动计算缺失的序列特征；CPU 全量编码和训练耗时较长。

## 模型

| 模型 | 用途 |
|---|---|
| **BaseModel** | 公开数据训练的专家与生成器；需经实验校准后才能预测 MIC。 |
| **FinalModel** | 校准后的模型，默认用于预测、继续校准及候选评分。 |

通过 `--model BaseModel` 或 `--model path/to/model.pt` 指定其他起始模型。自定义权重需附带 `scorer.json`。校准默认保存新模型，增加 `--update-final` 才会将其设为默认模型。

## 输入与输出

**预测：** 支持 FASTA、CSV、TSV、XLSX（首个工作表）。表格必需 `sequence` 列，可选 `id` 列。序列使用 20 种标准氨基酸，不需要菌株信息。

**校准：** 支持 CSV、TSV、XLSX，每行一条观测：

| 列名 | 必需 | 内容 |
|---|---|---|
| `sequence` | 是 | 氨基酸序列 |
| `mic` | 是 | 输入 MIC，单位 μg/mL，例如 `8` 或 `>128` |
| `id` | 否 | 多肽编号 |
| `strain` | 否 | 实验菌株 |
| `round` | 否 | 实验轮次 |

默认校准标签为跨菌株最优 MIC **除以 10**。请勿提前缩放输入；使用 `--mic-divisor 1` 可保留原始尺度。标签来源和解释见[模型与数据说明](docs/MODELS_AND_DATA-Zh.md)。

| 命令 | 主要输出 |
|---|---|
| `predict` | `predictions.csv`：`id`、`sequence`、`predicted_mic` |
| `calibrate` | `model.pt`、`scorer.json`、`calibration.csv` |
| `generate` | `candidates.csv`、`candidates.fasta` |
| `train` | 专家权重、生成器及 `scorer.json` |

测试示例位于 [`demo/`](demo/README.md)，空白校准模板为 [`templates/experiments.csv`](templates/experiments.csv)。训练参数包括 `--epochs`、`--batch-size`、`--learning-rate` 和 `--seed`。预测或生成时增加 `--details` 可输出专家分数。

## 模型与数据说明

FinalModel 使用调整后的 MIC 标签及排名偏好拟合，输出属于模型估计，不是新增实验测量。候选生成使用预训练自回归生成器采样、过滤，再由所选模型排序，不会在每次请求时重新训练生成器。Demo 序列参与过校准，不是独立测试集。

资产版本、训练输入、候选表格和复现细节见[模型与数据说明](docs/MODELS_AND_DATA-Zh.md)。上游模型与数据集保留各自许可。

## 测试

```bash
python -m pytest -q
```
