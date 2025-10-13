"""
融合策略实现

本模块提供特征融合和优先级排序的核心算法实现。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureFusionStrategy:
    """特征融合策略"""
    
    def __init__(self, fusion_method: str = 'weighted_average'):
        """
        初始化特征融合策略
        
        Args:
            fusion_method: 融合方法 ('weighted_average', 'max', 'min', 'ensemble')
        """
        self.fusion_method = fusion_method
        self.scaler = MinMaxScaler()
        
    def normalize_scores(self, scores: pd.DataFrame) -> pd.DataFrame:
        """
        标准化评分
        
        Args:
            scores: 评分数据框
            
        Returns:
            pd.DataFrame: 标准化后的评分
        """
        normalized_scores = scores.copy()
        
        # 对每列进行0-1标准化
        for col in scores.columns:
            if col != 'feature_name':
                normalized_scores[col] = self.scaler.fit_transform(scores[[col]])
        
        return normalized_scores
    
    def weighted_average_fusion(self, scores: pd.DataFrame, weights: Dict[str, float] = None) -> pd.Series:
        """
        加权平均融合
        
        Args:
            scores: 标准化评分数据框
            weights: 权重字典
            
        Returns:
            pd.Series: 融合后的评分
        """
        if weights is None:
            # 等权重
            weights = {col: 1.0 for col in scores.columns if col != 'feature_name'}
        
        # 计算加权平均
        weighted_scores = pd.Series(index=scores.index, dtype=float)
        
        for idx, row in scores.iterrows():
            weighted_score = 0
            total_weight = 0
            
            for col, weight in weights.items():
                if col in row and not pd.isna(row[col]):
                    weighted_score += row[col] * weight
                    total_weight += weight
            
            if total_weight > 0:
                weighted_scores[idx] = weighted_score / total_weight
            else:
                weighted_scores[idx] = 0
        
        return weighted_scores
    
    def ensemble_fusion(self, scores: pd.DataFrame, methods: List[str] = None) -> pd.Series:
        """
        集成融合
        
        Args:
            scores: 标准化评分数据框
            methods: 融合方法列表
            
        Returns:
            pd.Series: 融合后的评分
        """
        if methods is None:
            methods = ['weighted_average', 'max', 'min']
        
        ensemble_scores = []
        
        for method in methods:
            if method == 'weighted_average':
                score = self.weighted_average_fusion(scores)
            elif method == 'max':
                score = scores.select_dtypes(include=[np.number]).max(axis=1)
            elif method == 'min':
                score = scores.select_dtypes(include=[np.number]).min(axis=1)
            else:
                continue
            
            ensemble_scores.append(score)
        
        # 对多个方法的结果进行平均
        if ensemble_scores:
            final_score = pd.concat(ensemble_scores, axis=1).mean(axis=1)
        else:
            final_score = pd.Series(index=scores.index, dtype=float)
        
        return final_score
    
    def fuse_features(self, scores: pd.DataFrame, weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        特征融合主函数
        
        Args:
            scores: 评分数据框
            weights: 权重字典
            
        Returns:
            pd.DataFrame: 融合结果
        """
        # 标准化评分
        normalized_scores = self.normalize_scores(scores)
        
        # 根据融合方法计算最终评分
        if self.fusion_method == 'weighted_average':
            final_scores = self.weighted_average_fusion(normalized_scores, weights)
        elif self.fusion_method == 'max':
            final_scores = normalized_scores.select_dtypes(include=[np.number]).max(axis=1)
        elif self.fusion_method == 'min':
            final_scores = normalized_scores.select_dtypes(include=[np.number]).min(axis=1)
        elif self.fusion_method == 'ensemble':
            final_scores = self.ensemble_fusion(normalized_scores)
        else:
            raise ValueError(f"不支持的融合方法: {self.fusion_method}")
        
        # 创建结果数据框
        result_df = scores.copy()
        result_df['fused_score'] = final_scores
        result_df['rank'] = result_df['fused_score'].rank(ascending=False)
        
        # 按融合评分排序
        result_df = result_df.sort_values('fused_score', ascending=False)
        
        logger.info(f"特征融合完成，融合方法: {self.fusion_method}")
        return result_df


class PriorityRankingStrategy:
    """优先级排序策略"""
    
    def __init__(self, ranking_method: str = 'score_based'):
        """
        初始化优先级排序策略
        
        Args:
            ranking_method: 排序方法 ('score_based', 'risk_level', 'multi_criteria')
        """
        self.ranking_method = ranking_method
        
    def score_based_ranking(self, df: pd.DataFrame, score_column: str = 'fused_score') -> pd.DataFrame:
        """
        基于评分的排序
        
        Args:
            df: 数据框
            score_column: 评分列名
            
        Returns:
            pd.DataFrame: 排序后的数据框
        """
        result_df = df.copy()
        result_df['priority_rank'] = result_df[score_column].rank(ascending=False)
        result_df = result_df.sort_values('priority_rank')
        
        return result_df
    
    def risk_level_ranking(self, df: pd.DataFrame, risk_levels: Dict[str, int] = None) -> pd.DataFrame:
        """
        基于风险等级的排序
        
        Args:
            df: 数据框
            risk_levels: 风险等级映射
            
        Returns:
            pd.DataFrame: 排序后的数据框
        """
        if risk_levels is None:
            risk_levels = {'high': 3, 'medium': 2, 'low': 1}
        
        result_df = df.copy()
        
        # 添加风险等级权重
        result_df['risk_weight'] = result_df.get('risk_level', 'medium').map(risk_levels)
        result_df['weighted_score'] = result_df['fused_score'] * result_df['risk_weight']
        
        # 基于加权评分排序
        result_df['priority_rank'] = result_df['weighted_score'].rank(ascending=False)
        result_df = result_df.sort_values('priority_rank')
        
        return result_df
    
    def multi_criteria_ranking(self, df: pd.DataFrame, criteria_weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        多准则排序
        
        Args:
            df: 数据框
            criteria_weights: 准则权重
            
        Returns:
            pd.DataFrame: 排序后的数据框
        """
        if criteria_weights is None:
            criteria_weights = {
                'fused_score': 0.4,
                'confidence': 0.3,
                'frequency': 0.2,
                'impact': 0.1
            }
        
        result_df = df.copy()
        
        # 计算多准则评分
        multi_criteria_score = pd.Series(index=df.index, dtype=float)
        
        for idx, row in df.iterrows():
            score = 0
            for criterion, weight in criteria_weights.items():
                if criterion in row and not pd.isna(row[criterion]):
                    score += row[criterion] * weight
            multi_criteria_score[idx] = score
        
        result_df['multi_criteria_score'] = multi_criteria_score
        result_df['priority_rank'] = result_df['multi_criteria_score'].rank(ascending=False)
        result_df = result_df.sort_values('priority_rank')
        
        return result_df
    
    def rank_features(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        特征排序主函数
        
        Args:
            df: 数据框
            **kwargs: 额外参数
            
        Returns:
            pd.DataFrame: 排序结果
        """
        if self.ranking_method == 'score_based':
            return self.score_based_ranking(df, **kwargs)
        elif self.ranking_method == 'risk_level':
            return self.risk_level_ranking(df, **kwargs)
        elif self.ranking_method == 'multi_criteria':
            return self.multi_criteria_ranking(df, **kwargs)
        else:
            raise ValueError(f"不支持的排序方法: {self.ranking_method}")


def comprehensive_feature_fusion_and_ranking(
    association_scores: pd.DataFrame,
    model_scores: pd.DataFrame,
    weights: Dict[str, float] = None
) -> pd.DataFrame:
    """
    综合特征融合和排序
    
    Args:
        association_scores: 关联规则评分
        model_scores: 模型评分
        weights: 权重字典
        
    Returns:
        pd.DataFrame: 综合排序结果
    """
    # 特征融合
    fusion_strategy = FeatureFusionStrategy(fusion_method='weighted_average')
    fused_features = fusion_strategy.fuse_features(association_scores, weights)
    
    # 优先级排序
    ranking_strategy = PriorityRankingStrategy(ranking_method='multi_criteria')
    final_ranking = ranking_strategy.rank_features(fused_features)
    
    logger.info("综合特征融合和排序完成")
    return final_ranking
