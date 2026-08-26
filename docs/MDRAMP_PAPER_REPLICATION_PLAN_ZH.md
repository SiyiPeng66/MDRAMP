# MDRAMP 论文设计复刻计划书

## 1. 目标与边界

本计划的目标是将当前仓库从早期原型逐步完善为能够执行、审计并复核 `NCS_manuscript_V7.docx` 和配套补充材料所描述方法学的实现。

这里的“复刻”指：

- 论文中的数据规则、模型结构、训练目标、状态更新、候选选择和评分公式都有明确代码实现；
- 每一轮的可见数据和冻结边界可以由程序验证；
- 论文表格和图所需的中间数据能够从原始输入和冻结 checkpoint 重新生成；
- 论文中的结果不会通过硬编码或手工 CSV 注入；
- 对于仓库中无法提供的原始实验数据、外部模型权重或历史 checkpoint，明确标记为不可复核部分，不伪造结果。

当前仓库是早期版本。本计划保留其已有的数据解析、ESM3 封装、外部教师接口和基础门控积分器，采用渐进式替换方式，不把当前简化实现直接视为论文最终实现。

## 2. 复刻原则

1. **方法学优先**：以主文 Methods、Supplementary Methods 及表格中的定义为准，README 和旧代码仅作为实现线索。
2. **状态隔离优先**：任何当前轮标签都不得进入当前轮特征、模型、阈值、候选池、排序或 reserve list。
3. **证据与概率分离**：`p_cons` 是一致性排序证据，`A_t` 是 MIC 活性优先级，evidential probability 只用于 post-campaign 校准分析。
4. **冻结优先**：ESM3、蛋白 panel、Expert backbone、gate constants、scaler 和 selection policy 必须具备可追溯的冻结版本。
5. **可复核优先**：每个核心公式、数据契约和状态转换都必须有单元测试或 replay 测试。
6. **不虚构数据**：缺失的历史实验 ledger、外部权重或论文专用输入只能报告为缺口，不能用模拟值冒充论文结果。

## 3. 目标架构

```text
raw public/private/target data
        |
        v
curation + background construction + frozen protein panel
        |
        v
frozen ESM3 embeddings + frozen state-specific feature scaler
        |
        +--> AMP Prior Expert ensemble
        +--> Membrane-Perturbation Expert ensemble
        +--> Protein-Interaction Coverage Expert ensemble
        |          |
        |          +--> pair score -> top-five attention + Noisy-OR
        |
        +--> uncertainty-adjusted evidence
                    |
                    v
              p_cons(x)
                    |
        frozen Expert means + ESM3 + physicochemical features
                    |
                    v
             MIC regression head
                    |
                    v
              A_t(x), R_t(x)
                    |
                    v
        guardrails + mixed policy + reserve list + experiment ledger
                    |
                    v
                next model state
```

## 4. 阶段计划

### 阶段 0：基线冻结与方法学映射

**目标**：在改代码前建立论文-代码差距的可审计基线。

**工作内容**：

- 固化主文和补充材料版本、文件 hash 和读取日期；
- 建立论文方法条款编号，例如 `M-Data-01`、`M-Expert-03`、`M-Score-02`；
- 为每条条款登记状态：`implemented`、`partial`、`missing`、`blocked_by_external_input`；
- 记录当前仓库版本、工作区已有修改和当前 import smoke test；
- 明确外部依赖：ESM3、Pore-Forming、TPepPro、HemoPI-2、历史实验 ledger。

**主要产物**：

- `docs/method_traceability_matrix.csv`；
- `docs/replication_gap_report.md`；
- 基线测试报告。

**验收标准**：每一条论文方法要求都能定位到文档段落和预期源码/测试位置。

### 阶段 1：数据契约、背景集与 prequential 隔离

**目标**：先解决数据正确性和泄漏风险，使后续模型训练有可靠输入。

**工作内容**：

- 完善公开数据来源、清洗、去重和 cluster split；
- 加入 UniProt/Swiss-Prot background 构造及排除规则；
- 实现 peptide/protein exact duplicate、close homology 和 cross-partition exclusion；
- 构建并冻结 21,445 条工作蛋白和 multi-target protein panel；
- 扩展 schema：实验 endpoint、MIC、censor flag、round visibility、selection category、parent scaffold；
- 实现 state-specific data cutoff 和 label-release manifest；
- 实现 reserve list、replacement record 和 challenge-set exclusion；
- 建立 state-aware scaler，只用当前状态可见数据拟合。

**主要模块**：

- `src/data/parse_public.py`
- `src/data/parse_private_round.py`
- `src/data/parse_targets.py`
- 新增 `src/data/background.py`
- 新增 `src/data/protein_panel.py`
- 新增 `src/data/ledger.py`
- 新增 `src/loop/prequential_guard.py`
- 新增 `src/features/scaler.py`

**验收标准**：

- 公开数据统计可复现：80,725 -> 55,493 -> 23,573 -> 16,489；
- split 数量可复现：19,093 / 2,353 / 2,127；
- 当前轮标签对当前轮训练、排序、阈值和候选生成的访问会被测试拦截；
- 所有 state artifact 都记录数据 cutoff 和输入 hash。

### 阶段 2：ESM3 表征与特征冻结

**目标**：保证所有数据域使用同一 ESM3 表征和特征标准化流程。

**工作内容**：

- 固化 ESM3 checkpoint alias、版本、model hash 和 embedding metadata；
- 统一 peptide、target protein、candidate、comparator 的编码接口；
- 保留 residue/global embedding，并明确 global embedding 生成规则；
- 接入 state-specific scaler；
- 实现 representation-space cosine distance 和 diversity matrix；
- 实现 source-versus-tested representation discriminator 及 post-campaign density-ratio replay。

**主要模块**：

- `src/features/esm3_encoder.py`
- `src/features/build_esm3_embeddings.py`
- `src/features/embedding_store.py`
- 新增 `src/features/representation_shift.py`

**验收标准**：相同输入、相同 checkpoint 和相同 metadata 在不同运行中得到相同 embedding hash；不同 state 不能意外覆盖 frozen embedding。

### 阶段 3：三类 Expert 及五成员 ensemble

**目标**：用论文架构替换当前统一 MLP 简化版。

**AMP Prior Expert**：

- 512–256 GELU 主干；
- non-negative PU branch；
- AMP-manifold compatibility branch；
- curated positives + length-matched operational unlabelled background；
- 不把 background 当 confirmed negative。

**Membrane Expert**：

- 512–256 sequence branch；
- 128–64 physicochemical branch；
- feature-wise gate 融合至 256 维；
- primary sigmoid head + 四个 masked auxiliary heads；
- 缺失 annotation 不产生负梯度；
- PF-Teacher 连续 soft supervision。

**Protein-Interaction Coverage Expert**：

- peptide/protein 各自投影至 256 维；
- absolute difference、element-wise product、32 个理化特征；
- rank-64 bilinear interaction；
- 512–256 GELU pair head；
- sparse top-five attention + Noisy-OR。

**共同要求**：

- 每个 Expert 五个成员；
- seeds 固定为 11、23、37、51、73；
- 输出 ensemble mean、standard deviation、member outputs；
- ESM3 保持 frozen；
- 保存 member-level checkpoint、ensemble manifest 和参数量审计。

**主要模块**：

- `src/models/expert_modules.py`
- 新增 `src/models/ensemble.py`
- 新增 `src/models/amp_prior.py`
- 新增 `src/models/membrane_expert.py`
- 新增 `src/models/interaction_expert.py`
- `src/train/pretrain_public.py`
- `src/train/train_target_affinity.py`
- 新增对应训练 loss 和 dataset 模块。

**验收标准**：结构、输入维度、参数量、seed、member 数量和输出统计都能自动审计；所有 Expert 的训练目标与论文一致。

### 阶段 4：一致性积分与不确定性

**目标**：实现论文中的 uncertainty-adjusted `p_cons`，并保持与 calibrated probability 分离。

**工作内容**：

- 实现 `S_j = clip(mu_j - kappa*sigma_j, 0, 1)`；
- 固定 `kappa=1.0`、`tau_prior=0.5`、`beta=10`、`lambda_syn=0.3`；
- 实现 `kappa={0,0.5,1.5}` sensitivity；
- 统一 `S_int`/protein-interaction coverage 命名；
- 输出 `S_prior`、`S_mem`、`S_int`、`E_comp`、`E_syn`、`E_mech`、`p_cons`；
- 明确禁止把 `p_cons` 当绝对活动概率。

**主要模块**：

- `src/models/gated_integrator.py`
- `src/models/mdramp_system.py`
- 新增 `src/score/integration_audit.py`

**验收标准**：NumPy、PyTorch 和端到端结果在容差范围内一致；公式边界值和 uncertainty penalty 有单元测试。

### 阶段 5：MIC regression 与 R0–R3 部署排序

**目标**：实现论文真正使用的候选排序，而不是只按 `p_cons` 排名。

**工作内容**：

- 新增 16-unit MIC head；
- 输入 frozen ESM3 + 32 descriptors + 三个 Expert means；
- 预测 `log2(MIC)`；
- uncensored 使用 Huber loss；
- `>256` 使用 right-censored one-sided Huber loss；
- Expert 输出 stop-gradient；
- R0 初始化 MIC head；
- 后续 state 只用新释放的定量 MIC 更新 MIC head；
- 固定 `A_t=sigmoid(5-y_hat_t)`；
- 固定部署排序 `R_t=p_cons*A_t`；
- 输出并记录 `p_cons`、`A_t`、`R_t` 和预测 MIC。

**主要模块**：

- 新增 `src/models/mic_regressor.py`
- 新增 `src/train/train_mic_head.py`
- 修改 `src/train/finetune_round.py`
- 修改 `src/score/score_candidates.py`

**验收标准**：同一 frozen backbone 下，更新 MIC label 只能改变 `A_t` 和 `R_t`，不能改变 Expert 参数或 `p_cons`；censored loss 有专门数值测试。

### 阶段 6：候选生成、过滤和混合选择

**目标**：复刻从 proposal pool 到实验队列的完整决策流程。

**工作内容**：

- 每个 state 生成/导入 200 条 proposal；
- 实现标准残基、长度、重复、homology、合成风险过滤；
- 实现 net charge、hydrophobic fraction、single-AA fraction 约束；
- 实现最小 pairwise ESM3 cosine distance 0.10；
- 实现 exploitation、uncertainty-boundary、embedding-diverse、novel/synthesis-feasible 混合 policy；
- 固化 selection quota、reserve list 和 replacement order；
- 实现 R0/R1/R2=30、R3=60、R4/R5=10 的 budget；
- 生成可审计 selection manifest。

**主要模块**：

- 新增 `src/selection/filters.py`
- 新增 `src/selection/mixed_policy.py`
- 新增 `src/selection/reserve_list.py`
- 新增 `src/selection/selection_manifest.py`

**验收标准**：任何入选 candidate 都能追溯到 proposal、过滤原因、score、category、reserve position 和 frozen timestamp。

### 阶段 7：R4/R5 Pep-B 定向优化和安全性约束

**目标**：实现论文中从 Pep-B0 到 Pep-B18 的第二阶段设计。

**工作内容**：

- 实现 constrained genetic algorithm；
- 固定 population、tournament、elitism、crossover、mutation、edit-event 参数；
- 固定五个优化 seeds：101、211、307、401、503；
- 实现 parent-length 和 edit-distance 限制；
- R4 只用 antibacterial activity recovery；
- R5 在 antibacterial scoring 之后接入 HemoPI-2 risk ranking；
- haemolysis 不得进入 `p_cons` 或 antibacterial training target；
- 实现 activity–safety Pareto selection。

**主要模块**：

- 新增 `src/optimization/pep_b_genetic.py`
- 新增 `src/optimization/mutation_ops.py`
- 新增 `src/external/hemopi_wrapper.py`
- 新增 `src/selection/pareto.py`

**验收标准**：R4/R5 的 activity model、safety model 和最终 selection decision 三者可分开复核；缺失 HemoPI 权重时明确阻断，而不是静默替代。

### 阶段 8：Post-campaign evidential calibration

**目标**：实现论文中只用于 retrospective analysis 的 evidential 模块。

**工作内容**：

- 实现 positive/negative evidence 到 Beta 参数的转换；
- 实现 vacuity、reliability discount 和 fused probability；
- 计算 ECE、Brier、NLL、reliability curve；
- 实现 shifted-set vacuity stress test；
- 明确该模块不参与 R0–R5 selection。

**主要模块**：

- 新增 `src/models/evidential.py`
- 新增 `src/score/calibration.py`
- 新增 `src/score/vacuity_analysis.py`

**验收标准**：prospective ranking replay 与 post-campaign calibration replay 使用不同输入和不同 artifact，程序层面不能混用。

### 阶段 9：结果重建、图表数据和最终审计

**目标**：验证代码是否可以从输入数据重建论文方法相关结果。

**工作内容**：

- 生成 R0–R3 prospective ledger 和 fixed challenge-set evaluation；
- 生成 hit rate、AUPRC、average precision、precision@k、top-k enrichment；
- 生成 Expert ablation、teacher control、protein-panel control、cold-split 分析；
- 生成 calibration/vacuity/reliability 分析；
- 生成 R4/R5 activity-safety 分析；
- 每个图表使用机器生成的数据文件，不手工录入数值；
- 建立端到端 `reproduce_all` 流程和 dry-run 模式。

**主要模块**：

- 新增 `src/evaluation/`
- 新增 `src/reports/`
- 新增 `scripts/reproduce_all.py`
- 新增 `tests/`

**验收标准**：

- 所有表格和图所需数据能从 ledger/checkpoint 重新计算；
- 结果文件带有 input hash、state name、code version 和 random seed；
- 任何无法复核的论文结果被显式标注为 `blocked` 或 `not_reproducible_from_repository`。

## 5. 建议开发顺序

实际开发按以下顺序进行：

1. 阶段 0：建立追踪矩阵和基线。
2. 阶段 1：先完成数据契约、背景集和 prequential guard。
3. 阶段 2：冻结 embedding/scaler/panel artifact。
4. 阶段 3：实现三类 Expert 和五成员 ensemble。
5. 阶段 4：接入 uncertainty-adjusted `p_cons`。
6. 阶段 5：实现 MIC head 和 `R_t`，替换当前 `p_cons`-only ranking。
7. 阶段 6：实现 proposal/filter/mixed selection。
8. 阶段 7：实现 R4/R5 和 HemoPI 接口。
9. 阶段 8：实现 retrospective evidential analysis。
10. 阶段 9：完成 replay、图表数据和最终审计。

不建议在阶段 3 之前开发 R4/R5，也不建议在 MIC head 和数据隔离完成之前声称 R0–R3 已复刻。

## 6. 测试策略

### 单元测试

- sequence cleaning、identity、cluster split；
- physicochemical descriptors；
- embedding alignment；
- pair feature shapes；
- Noisy-OR；
- uncertainty-adjusted integration；
- censored Huber loss；
- MIC-to-activity mapping；
- GA mutation/crossover constraints；
- Pareto filtering；
- evidential Beta/vacuity calculations。

### 契约测试

- 每个 CSV/NPZ/checkpoint 的 schema；
- public/private/target ID 对齐；
- target panel completeness；
- teacher score coverage；
- ensemble member count and seed；
- state artifact completeness。

### 泄漏测试

- 当前轮 label 不可见；
- challenge set 不进入拟合；
- scaler 不读取未来数据；
- MIC head 不反向更新 Expert；
- HemoPI 不进入 antibacterial backbone；
- comparator labels 不提前进入 R4 更新。

### Replay 测试

- 固定 seed 后重复运行结果一致；
- R0–R3 状态转换可重放；
- frozen challenge set 在所有 checkpoint 上保持不变；
- 删除派生输出后可从原始 artifact 重新生成。

## 7. 外部依赖与阻断条件

以下条件缺失时不能宣称“完整论文复刻”：

- ESM3 checkpoint 和可用运行环境缺失；
- Pore-Forming 权重或 PF-Teacher 版本缺失；
- TPepPro 模型、pair 输入或可转换输出缺失；
- HemoPI-2 权重缺失；
- 论文中使用的历史 prospective ledger 缺失；
- fixed challenge set、protein panel 或历史 checkpoint 缺失。

遇到阻断时，代码应：

- 显式抛出缺失依赖错误；
- 记录缺失 artifact 名称和预期 schema；
- 允许结构/单元测试继续运行；
- 不使用未经声明的替代模型生成“论文结果”。

## 8. 最终完成定义

只有以下条件全部满足，项目才可标记为“论文方法学复刻完成”：

- 论文条款追踪矩阵无未解释的 `missing`；
- 所有 `blocked` 项都有外部输入说明；
- Expert、MIC head、gate、selection、R4/R5 和 evidential 模块均有实现；
- R0–R5 state artifact 可创建、加载、校验和重放；
- 当前轮 label isolation 有自动化测试；
- 论文核心公式有数值一致性测试；
- 表格和图数据可以从输入重新生成；
- 代码不会把 `p_cons`、`A_t`、`R_t` 或 evidential probability 混为同一指标；
- 输出文件含完整 provenance；
- 对缺失外部数据的部分明确报告不可复核，而不是补造结果。

## 9. 第一轮开发任务

下一轮开始时，优先执行以下工作：

1. 建立 `docs/method_traceability_matrix.csv`；
2. 添加基础测试框架和当前版本 baseline tests；
3. 扩展数据 schema 与 state manifest；
4. 实现 prequential label-isolation guard；
5. 实现 MIC head 的数据接口和 loss 单测；
6. 在不改变当前旧代码入口的前提下，增加新版本模块入口；
7. 用小型 fixture 数据完成端到端 dry-run。

完成上述任务后，再进入三类 Expert 的结构替换和 ensemble 化。
