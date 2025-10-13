# 铁路风险分析项目

## 项目概述

本项目旨在通过数据挖掘和机器学习技术，对铁路运输风险进行综合分析，识别关键风险因素，并提供优先级排序和风险评估。

## 项目结构

```
railway_risk_analysis/
├── data/                           # 数据目录
│   ├── raw/                        # 原始数据（FRA REAIR报告等）
│   └── processed/                   # 预处理后的数据
├── notebooks/                      # Jupyter Notebooks
│   ├── 01_data_acquisition_eda.ipynb
│   ├── 02_preprocessing_llm_feature_engineering.ipynb
│   ├── 03_regional_analysis_special_state_selection.ipynb
│   ├── 04_association_rule_mining.ipynb
│   ├── 05_predictive_modeling_feature_importance.ipynb
│   ├── 06_feature_fusion_prioritization.ipynb
│   ├── 07_evaluation_validation.ipynb
│   └── scratchpad.ipynb
├── src/                           # 源代码模块
│   ├── data_loader.py             # 数据加载和清洗
│   ├── text_processor.py          # 文本处理（LLM集成）
│   ├── feature_engineer.py        # 特征工程
│   ├── association_miner.py       # 关联规则挖掘
│   ├── model_trainer.py           # 模型训练
│   ├── fusion_strategy.py         # 融合策略
│   └── utils.py                   # 工具函数
├── models/                        # 训练好的模型
├── results/                       # 分析结果
│   ├── visualizations/           # 图表和可视化
│   └── reports/                  # 分析报告
├── config/                        # 配置文件
├── .gitignore
├── README.md
└── requirements.txt
```

## 分析流程

### 阶段1 (S1): 数据获取与预处理
- **S1初期**: 数据获取与初步探索性数据分析
- **S1中后期**: 数据预处理、LLM应用和特征工程
- **S1后期**: 区域分析及"特殊州"选择

### 阶段2 (S2): 模式发现与建模
- **S2.1**: 关联规则挖掘
- **S2.2**: 预测模型训练与特征重要性提取
- **S2.3**: 特征融合与优先级排序（核心部分）

### 阶段3 (S3): 结果评估与验证
- 结果评估与验证

## 主要功能

### 数据获取与处理
- 支持多种数据格式（CSV、Excel等）
- 自动数据清洗和预处理
- 数据质量验证

### 文本分析
- LLM集成进行文本分析
- 关键词提取
- 情感分析

### 特征工程
- 时间特征创建
- 分类特征编码
- 数值特征变换
- 风险相关特征

### 关联规则挖掘
- 频繁项集发现
- 关联规则生成
- 模式分析

### 机器学习建模
- 多种算法支持（随机森林、梯度提升、逻辑回归）
- 特征重要性分析
- 模型性能评估

### 特征融合与排序
- 多种融合策略
- 优先级排序算法
- 综合风险评估

## 安装和使用

### 环境要求
- Python 3.8+
- Jupyter Notebook

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行分析
1. 将数据文件放入 `data/raw/` 目录
2. 按顺序运行 notebooks 中的分析流程
3. 查看 `results/` 目录中的分析结果

## 配置说明

在 `config/` 目录中配置：
- LLM API密钥
- 模型超参数
- 数据源路径
- 输出设置

## 注意事项

- 确保数据文件格式正确
- 根据实际需求调整模型参数
- 定期备份重要结果
- 注意数据隐私和安全

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 许可证

本项目采用 MIT 许可证。

