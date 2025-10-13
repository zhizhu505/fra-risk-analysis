"""
预测模型训练和重要性提取封装类/函数

本模块提供机器学习模型训练和特征重要性分析功能。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        初始化模型训练器
        
        Args:
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()
        
    def prepare_data(self, df: pd.DataFrame, target_column: str, feature_columns: List[str]) -> Tuple:
        """
        准备训练数据
        
        Args:
            df: 数据框
            target_column: 目标列名
            feature_columns: 特征列名列表
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        # 分离特征和目标
        X = df[feature_columns]
        y = df[target_column]
        
        # 处理缺失值
        X = X.fillna(X.mean())
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"数据准备完成，训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, **params) -> Dict[str, Any]:
        """
        训练随机森林模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            **params: 模型参数
            
        Returns:
            Dict: 模型结果
        """
        rf = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            random_state=self.random_state,
            **{k: v for k, v in params.items() if k not in ['n_estimators', 'max_depth']}
        )
        
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': range(X_train.shape[1]),
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['random_forest'] = feature_importance
        
        logger.info("随机森林模型训练完成")
        return {'model': rf, 'feature_importance': feature_importance}
    
    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray, **params) -> Dict[str, Any]:
        """
        训练梯度提升模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            **params: 模型参数
            
        Returns:
            Dict: 模型结果
        """
        gb = GradientBoostingClassifier(
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.1),
            max_depth=params.get('max_depth', 3),
            random_state=self.random_state,
            **{k: v for k, v in params.items() if k not in ['n_estimators', 'learning_rate', 'max_depth']}
        )
        
        gb.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': range(X_train.shape[1]),
            'importance': gb.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['gradient_boosting'] = feature_importance
        
        logger.info("梯度提升模型训练完成")
        return {'model': gb, 'feature_importance': feature_importance}
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray, **params) -> Dict[str, Any]:
        """
        训练逻辑回归模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            **params: 模型参数
            
        Returns:
            Dict: 模型结果
        """
        lr = LogisticRegression(
            random_state=self.random_state,
            **params
        )
        
        lr.fit(X_train, y_train)
        self.models['logistic_regression'] = lr
        
        # 特征重要性（系数）
        feature_importance = pd.DataFrame({
            'feature': range(X_train.shape[1]),
            'importance': np.abs(lr.coef_[0])
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['logistic_regression'] = feature_importance
        
        logger.info("逻辑回归模型训练完成")
        return {'model': lr, 'feature_importance': feature_importance}
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试目标
            
        Returns:
            Dict: 评估结果
        """
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # 评估指标
        metrics = {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'auc_score': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        }
        
        return metrics
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """
        获取特征重要性汇总
        
        Returns:
            pd.DataFrame: 特征重要性汇总
        """
        if not self.feature_importance:
            return pd.DataFrame()
        
        # 合并所有模型的特征重要性
        importance_dfs = []
        for model_name, importance_df in self.feature_importance.items():
            importance_df = importance_df.copy()
            importance_df['model'] = model_name
            importance_dfs.append(importance_df)
        
        if importance_dfs:
            combined_df = pd.concat(importance_dfs, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()

