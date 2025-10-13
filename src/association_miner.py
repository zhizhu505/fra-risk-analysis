"""
关联规则挖掘封装类/函数

本模块提供关联规则挖掘功能，用于发现数据中的模式和关联关系。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssociationRuleMiner:
    """关联规则挖掘器"""
    
    def __init__(self, min_support: float = 0.1, min_confidence: float = 0.5):
        """
        初始化关联规则挖掘器
        
        Args:
            min_support: 最小支持度
            min_confidence: 最小置信度
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = None
        self.rules = None
        
    def prepare_transaction_data(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        准备事务数据
        
        Args:
            df: 数据框
            columns: 用于关联规则挖掘的列名列表
            
        Returns:
            pd.DataFrame: 事务数据
        """
        # 将数据转换为事务格式
        transactions = []
        for _, row in df.iterrows():
            transaction = []
            for col in columns:
                if pd.notna(row[col]) and row[col] != '':
                    transaction.append(f"{col}_{row[col]}")
            if transaction:
                transactions.append(transaction)
        
        # 使用TransactionEncoder转换为二进制格式
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
        
        logger.info(f"事务数据准备完成，形状: {df_transactions.shape}")
        return df_transactions
    
    def find_frequent_itemsets(self, df_transactions: pd.DataFrame) -> pd.DataFrame:
        """
        发现频繁项集
        
        Args:
            df_transactions: 事务数据框
            
        Returns:
            pd.DataFrame: 频繁项集
        """
        self.frequent_itemsets = apriori(
            df_transactions, 
            min_support=self.min_support, 
            use_colnames=True
        )
        
        logger.info(f"发现 {len(self.frequent_itemsets)} 个频繁项集")
        return self.frequent_itemsets
    
    def generate_rules(self, frequent_itemsets: pd.DataFrame = None) -> pd.DataFrame:
        """
        生成关联规则
        
        Args:
            frequent_itemsets: 频繁项集，如果为None则使用已存储的
            
        Returns:
            pd.DataFrame: 关联规则
        """
        if frequent_itemsets is None:
            frequent_itemsets = self.frequent_itemsets
            
        if frequent_itemsets is None:
            raise ValueError("没有可用的频繁项集")
        
        self.rules = association_rules(
            frequent_itemsets, 
            metric="confidence", 
            min_threshold=self.min_confidence
        )
        
        logger.info(f"生成 {len(self.rules)} 条关联规则")
        return self.rules
    
    def analyze_rules(self, rules: pd.DataFrame = None) -> Dict[str, Any]:
        """
        分析关联规则
        
        Args:
            rules: 关联规则，如果为None则使用已存储的
            
        Returns:
            Dict: 规则分析结果
        """
        if rules is None:
            rules = self.rules
            
        if rules is None:
            raise ValueError("没有可用的关联规则")
        
        analysis = {
            'total_rules': len(rules),
            'avg_confidence': rules['confidence'].mean(),
            'avg_support': rules['support'].mean(),
            'avg_lift': rules['lift'].mean(),
            'high_confidence_rules': len(rules[rules['confidence'] > 0.8]),
            'high_lift_rules': len(rules[rules['lift'] > 2.0])
        }
        
        return analysis
    
    def get_top_rules(self, n: int = 10, metric: str = 'confidence') -> pd.DataFrame:
        """
        获取top规则
        
        Args:
            n: 返回规则数量
            metric: 排序指标
            
        Returns:
            pd.DataFrame: top规则
        """
        if self.rules is None:
            raise ValueError("没有可用的关联规则")
        
        return self.rules.nlargest(n, metric)


def mine_railway_risk_patterns(df: pd.DataFrame, risk_columns: List[str]) -> Dict[str, Any]:
    """
    挖掘铁路风险模式
    
    Args:
        df: 数据框
        risk_columns: 风险相关列名列表
        
    Returns:
        Dict: 风险模式挖掘结果
    """
    miner = AssociationRuleMiner(min_support=0.05, min_confidence=0.6)
    
    # 准备事务数据
    df_transactions = miner.prepare_transaction_data(df, risk_columns)
    
    # 发现频繁项集
    frequent_itemsets = miner.find_frequent_itemsets(df_transactions)
    
    # 生成关联规则
    rules = miner.generate_rules(frequent_itemsets)
    
    # 分析规则
    analysis = miner.analyze_rules(rules)
    
    # 获取top规则
    top_rules = miner.get_top_rules(n=20, metric='confidence')
    
    return {
        'frequent_itemsets': frequent_itemsets,
        'rules': rules,
        'analysis': analysis,
        'top_rules': top_rules
    }
