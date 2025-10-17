# 铁路风险分析项目

## 项目概述

本项目旨在通过数据挖掘与机器学习技术，对铁路运输安全事件进行系统性分析，识别关键风险因素，并对特征进行融合与优先级排序，最终输出可视化结果与报告，辅助风险治理与资源配置决策。

## 项目结构

```
railway_risk_analysis/
├── config/                        # 配置模板与实际配置（复制模板）
│   └── config_template.yaml
├── data/
│   ├── raw/                       # 原始数据（例如 FRA REAIR CSV 等）
│   └── processed/                 # 预处理与中间结果
├── notebooks/                     # 交互式分析 Notebooks（按阶段组织）
│   ├── 01_data_acquisition_eda.ipynb
│   ├── 02_preprocessing_llm_feature_engineering.ipynb
│   ├── 03_regional_analysis_special_state_selection.ipynb
│   ├── 04_association_rule_mining.ipynb
│   ├── 05_predictive_modeling_feature_importance.ipynb
│   ├── 06_feature_fusion_prioritization.ipynb
│   └── 07_evaluation_validation.ipynb
├── src/                           # 源代码模块
│   ├── data_loader.py             # 数据加载、初清洗、日期转换与缺失值处理建议
│   ├── text_processor.py          # 文本清洗与列合并（LLM 预处理/集成接口）
│   ├── feature_engineer.py        # 时间/分类/数值特征创建与选择
│   ├── association_miner.py       # 频繁项集与关联规则挖掘（mlxtend）
│   ├── model_trainer.py           # 模型训练与评估（RF/GB/LogReg）
│   ├── fusion_strategy.py         # 特征融合与多准则排序
│   ├── top_states_analysis.py     # 州级事故统计、核心列选择与文本质量检查脚本
│   ├── text_quality_report.py     # 文本质量报告（可扩展）
│   └── utils.py                   # 通用工具
├── models/                        # 训练好的模型文件（可选）
├── results/
│   ├── visualizations/            # 可视化图件（示例：text_merged 长度直方图）
│   └── reports/                   # 分析报告（HTML/PDF/Markdown）
├── fra-risk-analysis/             # （可选）子目录：同名模板与说明
├── README.md
└── requirements.txt
```

## 环境要求

- Python 3.8+
- Jupyter Notebook / JupyterLab
- Windows PowerShell 或其他终端（本仓库在 Windows 10 上验证）

安装依赖：

```bash
pip install -r requirements.txt
```

如使用 LLM 能力（可选），请另外安装并配置相应 SDK（例如 `openai`/`transformers`/`torch`）。

## 配置说明

复制 `config/config_template.yaml` 为 `config/config.yaml`，并根据需要修改：

- data.raw_data_path / processed_data_path / output_path
- models.*（随机森林、梯度提升、逻辑回归等超参数）
- association_rules.*（min_support、min_confidence 等）
- feature_fusion.*（method、weights）
- priority_ranking.*（method、criteria_weights）
- llm.*（provider、api_key、model 等，可选）
- logging.*、visualization.*、output.*

## 数据准备

将原始数据（如 FRA REAIR 清洗前 CSV）放入 `data/raw/`。部分脚本（如 `src/top_states_analysis.py`）默认读取 `data/processed/fra_reair_data_cleaned.csv`，请确保该文件已存在或修改脚本中的路径。

## 快速开始（Windows PowerShell 示例）

- 运行州级统计与文本清洗工作流（含保存 `df_selected_features_cleaned.csv` 与 `text_merged` 质量检查）：

```powershell
# 在项目根目录执行
python .\src\top_states_analysis.py
```

该脚本将：
- 统计“State Name”事故数 Top-N 并打印
- 输出事故类型编码映射表
- 生成各州×事故类型编码的百分比交叉表
- 从清洗数据中选择核心特征列，批量文本清洗，合并生成 `text_merged`
- 将清洗和合并后的数据保存到 `data/processed/`
- 输出 `text_merged` 的缺失率、长度统计并保存长度直方图到 `results/visualizations/`

如需自定义输入路径、Top-N 或是否计入缺失，可直接在脚本顶部或 `__main__` 段落中修改参数。

## 脚本与模块用法示例

- 数据加载与日期特征：参见 `src/data_loader.py` 中的 `load_fra_reair_data`、`convert_fra_date_columns`、`create_date_features` 等函数。
- 缺失值处理与建议：`handle_fra_missing_values`、`suggest_missing_value_strategies`。
- 特征工程：`feature_engineer.py` 的 `create_time_features`、`create_categorical_features`、`create_numerical_features`、`select_features`。
- 关联规则挖掘：`association_miner.py` 的 `AssociationRuleMiner` 或 `mine_railway_risk_patterns`。
- 模型训练与评估：`model_trainer.py` 的 `ModelTrainer`（支持 RF/GB/LogReg 与 AUC/报告/混淆矩阵）。
- 特征融合与排序：`fusion_strategy.py` 的 `FeatureFusionStrategy` 与 `PriorityRankingStrategy`，以及 `comprehensive_feature_fusion_and_ranking`。

## Notebooks 工作流

Notebooks 按阶段组织（S1→S2→S3）：
- S1 数据获取与预处理：01、02、03
- S2 模式发现与建模：04、05、06
- S3 评估与验证：07

建议按编号顺序运行，并将输出保存到 `data/processed/` 与 `results/`。

## 结果与输出

- 中间/最终数据：`data/processed/`
- 模型与特征重要性：`models/`（当 `output.save_models: true`）
- 可视化图件：`results/visualizations/`
- 报告：`results/reports/`（受 `output.report_format` 控制）

## 常见问题

- CSV 大文件读取：已在 `data_loader.py` 中启用 `low_memory=False`；必要时可考虑分块读取。
- 列名差异：`top_states_analysis.py` 对核心列做了规范化匹配，若报缺失，请对照实际列名调整。
- 文件占用：脚本在写入失败时会自动添加时间戳另存。

## 许可证

本项目采用 MIT 许可证。

## src 模块说明

- `add_state_name.py`
  - 功能: 将源数据集中的州名列 `State Name` 补充到目标 CSV。优先按索引对齐；若行数不一致且存在公共键（如 `Date`/`Time`/`Accident Type` 组合），则基于键左连接合并。
  - 特点: 写入被占用时自动回退为带时间戳的新文件，避免写入失败。
  - 入口: 直接运行脚本可执行一次补充示例。

- `association_miner.py`
  - 功能: 关联规则挖掘（频繁项集与规则生成），基于 mlxtend 的 `apriori` 与 `association_rules`。
  - 主要内容: `AssociationRuleMiner` 类（准备事务数据、挖掘频繁项集、生成规则、规则分析、Top-N 获取）；`mine_railway_risk_patterns` 一体化封装。

- `data_loader.py`
  - 功能: 数据加载、初步清洗、日期处理与缺失值策略建议。
  - 主要内容: 通用加载 `load_railway_data`、专用 `load_fra_reair_data`；`clean_initial_data`、`validate_data_quality`；日期解析与派生（`convert_fra_date_columns`、`handle_fra_date_formats`、`create_date_features`）；缺失值处理 `handle_fra_missing_values` 与策略建议 `suggest_missing_value_strategies`。

- `feature_engineer.py`
  - 功能: 特征工程（时间/分类/数值特征创建与筛选）。
  - 主要内容: `create_time_features`、`create_categorical_features`、`create_numerical_features`、`select_features`，以及风险特征扩展点 `engineer_risk_features`。

- `fusion_strategy.py`
  - 功能: 特征融合与优先级排序策略，实现多路评分的融合与排序。
  - 主要内容: `FeatureFusionStrategy`（0-1 标准化、加权平均/最大值/最小值/集成融合，输出 `fused_score` 与 `rank`）；`PriorityRankingStrategy`（评分/风险等级/多准则）；`comprehensive_feature_fusion_and_ranking` 融合+排序一体化。

- `model_trainer.py`
  - 功能: 监督学习训练与评估，输出特征重要性。
  - 主要内容: `ModelTrainer` 提供数据准备（缺失填充、划分、标准化）、训练 RF/GB/LogReg、评估（分类报告、混淆矩阵、AUC）、重要性汇总。

- `text_processor.py`
  - 功能: 文本清洗、关键词提取、基础情感分析、LLM 集成占位；批量列清洗与多列合并；文本质量评估与直方图绘制辅助。
  - 主要内容: 基础/高级清洗（`preprocess_text`、`preprocess_text_advanced`）；批量清洗 `preprocess_text_columns`；合并列 `merge_text_columns`；质量评估与可视化（`compute_text_column_quality`、`save_length_histogram`、中文字体配置）。
  - 注意: 若发现 Git 合并标记，请尽快清理以保持稳定。

- `text_quality_report.py`
  - 功能: 命令行动作的文本质量报告工具。
  - 用法: 传入 `--csv` 与 `--column`（默认 `text_merged`），打印缺失率、长度统计、Top-10 词频、随机样本，并保存长度直方图到 `results/visualizations/`。

- `top_states_analysis.py`
  - 功能: 州级统计与文本清洗示例工作流。
  - 主要内容: `compute_top_states` 统计 Top-N 州；`get_accident_type_mapping` 事故类型映射；`select_core_features` 核心列筛选（带规范化匹配）；主流程执行清洗、合并 `text_merged`、质量检查、图件输出与样例打印。
  - 用法: 在项目根目录直接运行，作为数据理解与文本清洗的起步脚本。

- `utils.py`
  - 功能: 通用工具与可视化辅助。
  - 主要内容: 绘图风格设定；结果保存/加载（CSV/JSON/Pickle/Excel）；摘要统计 `create_summary_statistics` 与质量校验 `validate_data_quality`；特征重要性条形图、相关性热力图；时间戳与目录保障。


