"""
特征工程函数

本模块提供特征工程相关的功能，包括特征创建、选择和转换。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_time_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    创建时间相关特征
    
    Args:
        df: 数据框
        date_column: 日期列名
        
    Returns:
        pd.DataFrame: 添加了时间特征的数据框
    """
    df = df.copy()
    
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        
        # 提取时间特征
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['dayofweek'] = df[date_column].dt.dayofweek
        df['quarter'] = df[date_column].dt.quarter
        
        logger.info("时间特征创建完成")
    
    return df


def create_categorical_features(df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    """
    创建分类特征
    
    Args:
        df: 数据框
        categorical_columns: 分类列名列表
        
    Returns:
        pd.DataFrame: 添加了分类特征的数据框
    """
    df = df.copy()
    
    for col in categorical_columns:
        if col in df.columns:
            # 创建独热编码
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            
            # 创建标签编码
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    
    logger.info("分类特征创建完成")
    return df


def create_numerical_features(df: pd.DataFrame, numerical_columns: List[str]) -> pd.DataFrame:
    """
    创建数值特征
    
    Args:
        df: 数据框
        numerical_columns: 数值列名列表
        
    Returns:
        pd.DataFrame: 添加了数值特征的数据框
    """
    df = df.copy()
    
    for col in numerical_columns:
        if col in df.columns:
            # 对数变换
            df[f'{col}_log'] = np.log1p(df[col])
            
            # 平方根变换
            df[f'{col}_sqrt'] = np.sqrt(df[col])
            
            # 标准化
            scaler = StandardScaler()
            df[f'{col}_scaled'] = scaler.fit_transform(df[[col]])
    
    logger.info("数值特征创建完成")
    return df


def select_features(df: pd.DataFrame, target_column: str, k: int = 10) -> List[str]:
    """
    特征选择
    
    Args:
        df: 数据框
        target_column: 目标列名
        k: 选择的特征数量
        
    Returns:
        List[str]: 选择的特征列表
    """
    # 分离特征和目标
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 选择数值特征
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_features]
    
    # 使用SelectKBest进行特征选择
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X_numeric, y)
    
    selected_features = [numeric_features[i] for i in selector.get_support(indices=True)]
    
    logger.info(f"选择了 {len(selected_features)} 个特征")
    return selected_features


def engineer_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建风险相关特征
    
    Args:
        df: 数据框
        
    Returns:
        pd.DataFrame: 添加了风险特征的数据框
    """
    df = df.copy()
    
    # 这里可以根据具体的风险分析需求创建特征
    # 例如：风险等级、风险类型、风险严重程度等
    
    logger.info("风险特征工程完成")
    return df

