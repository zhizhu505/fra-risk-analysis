"""
常用工具函数

本模块提供项目中常用的工具函数。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import pickle
from datetime import datetime
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_plotting_style():
    """设置绘图样式"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12


def save_results(data: Any, file_path: str, format: str = 'csv') -> None:
    """
    保存结果数据
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        format: 保存格式 ('csv', 'json', 'pickle', 'excel')
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if format == 'csv' and isinstance(data, pd.DataFrame):
        data.to_csv(file_path, index=False, encoding='utf-8-sig')
    elif format == 'json':
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif format == 'pickle':
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'excel' and isinstance(data, pd.DataFrame):
        data.to_excel(file_path, index=False)
    else:
        raise ValueError(f"不支持的保存格式: {format}")
    
    logger.info(f"结果已保存到: {file_path}")


def load_results(file_path: str, format: str = 'csv') -> Any:
    """
    加载结果数据
    
    Args:
        file_path: 文件路径
        format: 文件格式
        
    Returns:
        Any: 加载的数据
    """
    if format == 'csv':
        return pd.read_csv(file_path, encoding='utf-8-sig')
    elif format == 'json':
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif format == 'pickle':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif format == 'excel':
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的加载格式: {format}")


def create_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    创建数据摘要统计
    
    Args:
        df: 数据框
        
    Returns:
        Dict: 摘要统计信息
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        'categorical_summary': {}
    }
    
    # 分类变量摘要
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        summary['categorical_summary'][col] = {
            'unique_count': df[col].nunique(),
            'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
            'frequency': df[col].value_counts().head().to_dict()
        }
    
    return summary


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20, title: str = "特征重要性") -> None:
    """
    绘制特征重要性图
    
    Args:
        importance_df: 特征重要性数据框
        top_n: 显示前N个特征
        title: 图表标题
    """
    setup_plotting_style()
    
    # 选择前N个特征
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title(title)
    plt.xlabel('重要性')
    plt.ylabel('特征')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, title: str = "特征相关性热力图") -> None:
    """
    绘制相关性热力图
    
    Args:
        df: 数据框
        title: 图表标题
    """
    setup_plotting_style()
    
    # 计算相关性矩阵
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def generate_report_summary(results: Dict[str, Any]) -> str:
    """
    生成报告摘要
    
    Args:
        results: 分析结果字典
        
    Returns:
        str: 报告摘要
    """
    summary = f"""
# 铁路风险分析报告摘要
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 分析结果概览
"""
    
    for key, value in results.items():
        if isinstance(value, dict):
            summary += f"\n### {key}\n"
            for sub_key, sub_value in value.items():
                summary += f"- {sub_key}: {sub_value}\n"
        else:
            summary += f"- {key}: {value}\n"
    
    return summary


def validate_data_quality(df: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
    """
    验证数据质量
    
    Args:
        df: 数据框
        required_columns: 必需的列名列表
        
    Returns:
        Dict: 数据质量报告
    """
    quality_report = {
        'is_valid': True,
        'issues': [],
        'warnings': []
    }
    
    # 检查必需列
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            quality_report['is_valid'] = False
            quality_report['issues'].append(f"缺少必需列: {missing_columns}")
    
    # 检查空值
    null_counts = df.isnull().sum()
    high_null_columns = null_counts[null_counts > len(df) * 0.5].index.tolist()
    if high_null_columns:
        quality_report['warnings'].append(f"高空值列 (>50%): {high_null_columns}")
    
    # 检查重复行
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        quality_report['warnings'].append(f"发现 {duplicate_count} 行重复数据")
    
    # 检查数据类型
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) == 0:
        quality_report['warnings'].append("没有发现数值型列")
    
    return quality_report


def create_timestamp() -> str:
    """创建时间戳字符串"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def ensure_directory_exists(directory_path: str) -> None:
    """确保目录存在"""
    os.makedirs(directory_path, exist_ok=True)
    logger.info(f"目录已创建或存在: {directory_path}")

