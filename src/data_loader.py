"""
数据加载和初始清洗函数

本模块提供铁路风险分析项目的数据加载和初始清洗功能。
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_railway_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    加载铁路数据
    
    Args:
        file_path: 数据文件路径
        **kwargs: 传递给pandas读取函数的额外参数
        
    Returns:
        pd.DataFrame: 加载的数据框
    """
    try:
        if file_path.endswith('.csv'):
            # 对于大型CSV文件，使用chunking和low_memory=False
            data = pd.read_csv(file_path, low_memory=False, **kwargs)
        elif file_path.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
            
        logger.info(f"成功加载数据，形状: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        raise


def load_fra_reair_data(file_path: str) -> pd.DataFrame:
    """
    专门加载FRA REAIR数据
    
    Args:
        file_path: FRA REAIR数据文件路径
        
    Returns:
        pd.DataFrame: 加载的FRA数据框
    """
    try:
        logger.info("开始加载FRA REAIR数据...")
        data = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
        
        logger.info(f"FRA REAIR数据加载完成，形状: {data.shape}")
        logger.info(f"列数: {data.shape[1]}, 行数: {data.shape[0]}")
        
        return data
    except Exception as e:
        logger.error(f"加载FRA REAIR数据失败: {e}")
        raise


def clean_initial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    初始数据清洗
    
    Args:
        df: 原始数据框
        
    Returns:
        pd.DataFrame: 清洗后的数据框
    """
    # 删除完全空白的行和列
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    
    # 处理重复行
    df = df.drop_duplicates()
    
    logger.info(f"数据清洗完成，新形状: {df.shape}")
    return df


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    验证数据质量
    
    Args:
        df: 数据框
        
    Returns:
        Dict: 数据质量报告
    """
    quality_report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    
    return quality_report


def convert_fra_date_columns(df: pd.DataFrame, 
                           year_col: str = 'Accident Year', 
                           month_col: str = 'Accident Month',
                           day_col: str = None,
                           date_col: str = None) -> pd.DataFrame:
    """
    转换FRA REAIR数据中的日期列
    
    Args:
        df: 数据框
        year_col: 年份列名
        month_col: 月份列名
        day_col: 日期列名（可选）
        date_col: 完整日期列名（可选）
        
    Returns:
        pd.DataFrame: 转换后的数据框
    """
    df = df.copy()
    
    try:
        # 处理年份列
        if year_col in df.columns:
            logger.info(f"处理年份列: {year_col}")
            # 转换年份，处理可能的字符串格式
            df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
            
        # 处理月份列
        if month_col in df.columns:
            logger.info(f"处理月份列: {month_col}")
            # 转换月份，处理可能的字符串格式
            df[month_col] = pd.to_numeric(df[month_col], errors='coerce')
            
        # 处理日期列（如果存在）
        if day_col and day_col in df.columns:
            logger.info(f"处理日期列: {day_col}")
            df[day_col] = pd.to_numeric(df[day_col], errors='coerce')
            
        # 创建完整的日期列
        if year_col in df.columns and month_col in df.columns:
            logger.info("创建完整日期列...")
            
            # 创建日期字符串
            date_conditions = []
            date_choices = []
            
            # 情况1: 有年份和月份
            if year_col in df.columns and month_col in df.columns:
                mask = df[year_col].notna() & df[month_col].notna()
                date_conditions.append(mask)
                date_choices.append(
                    df[year_col].astype(str) + '-' + 
                    df[month_col].astype(str).str.zfill(2) + '-01'
                )
            
            # 情况2: 只有年份
            if year_col in df.columns:
                mask = df[year_col].notna() & df[month_col].isna()
                date_conditions.append(mask)
                date_choices.append(df[year_col].astype(str) + '-01-01')
            
            # 应用条件选择
            if date_conditions:
                df['accident_date'] = np.select(date_conditions, date_choices, default=pd.NaT)
                
                # 转换为datetime
                df['accident_date'] = pd.to_datetime(df['accident_date'], errors='coerce')
                
                # 创建额外的日期特征
                df['accident_year'] = df['accident_date'].dt.year
                df['accident_month'] = df['accident_date'].dt.month
                df['accident_quarter'] = df['accident_date'].dt.quarter
                df['accident_day_of_year'] = df['accident_date'].dt.dayofyear
                df['accident_weekday'] = df['accident_date'].dt.dayofweek
                df['accident_weekday_name'] = df['accident_date'].dt.day_name()
                
                logger.info("日期特征创建完成")
        
        # 处理现有的日期列（如果存在）
        if date_col and date_col in df.columns:
            logger.info(f"处理现有日期列: {date_col}")
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # 处理时间列（如果存在）
        time_cols = ['Time', 'time', 'TIME']
        for time_col in time_cols:
            if time_col in df.columns:
                logger.info(f"处理时间列: {time_col}")
                # 尝试解析时间
                df[f'{time_col}_parsed'] = pd.to_datetime(df[time_col], format='%H:%M', errors='coerce')
                if df[f'{time_col}_parsed'].notna().any():
                    df[f'{time_col}_hour'] = df[f'{time_col}_parsed'].dt.hour
                    df[f'{time_col}_minute'] = df[f'{time_col}_parsed'].dt.minute
                    df[f'{time_col}_time_of_day'] = df[f'{time_col}_parsed'].dt.hour.apply(
                        lambda x: 'Morning' if 6 <= x < 12 else 
                                 'Afternoon' if 12 <= x < 18 else 
                                 'Evening' if 18 <= x < 22 else 'Night'
                    )
        
        logger.info("日期列转换完成")
        return df
        
    except Exception as e:
        logger.error(f"日期列转换失败: {e}")
        raise


def handle_fra_date_formats(date_string: str) -> str:
    """
    处理FRA数据中的各种日期格式
    
    Args:
        date_string: 日期字符串
        
    Returns:
        str: 标准化的日期字符串
    """
    if pd.isna(date_string) or date_string == '':
        return pd.NaT
    
    date_string = str(date_string).strip()
    
    # 处理各种可能的格式
    formats_to_try = [
        '%Y-%m-%d',      # 2023-01-15
        '%Y-%m',         # 2023-01
        '%m/%d/%Y',      # 01/15/2023
        '%m-%d-%Y',      # 01-15-2023
        '%Y/%m/%d',      # 2023/01/15
        '%d/%m/%Y',      # 15/01/2023
        '%Y',            # 2023
        '%Y-%m-%d %H:%M:%S',  # 2023-01-15 14:30:00
    ]
    
    for fmt in formats_to_try:
        try:
            parsed_date = pd.to_datetime(date_string, format=fmt)
            return parsed_date.strftime('%Y-%m-%d')
        except:
            continue
    
    # 如果所有格式都失败，尝试pandas的自动解析
    try:
        parsed_date = pd.to_datetime(date_string)
        return parsed_date.strftime('%Y-%m-%d')
    except:
        logger.warning(f"无法解析日期: {date_string}")
        return pd.NaT


def create_date_features(df: pd.DataFrame, date_col: str = 'accident_date') -> pd.DataFrame:
    """
    从日期列创建额外的特征
    
    Args:
        df: 数据框
        date_col: 日期列名
        
    Returns:
        pd.DataFrame: 添加了日期特征的数据框
    """
    if date_col not in df.columns:
        logger.warning(f"日期列 {date_col} 不存在")
        return df
    
    df = df.copy()
    
    try:
        # 确保是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # 创建日期特征
        df[f'{date_col}_year'] = df[date_col].dt.year
        df[f'{date_col}_month'] = df[date_col].dt.month
        df[f'{date_col}_day'] = df[date_col].dt.day
        df[f'{date_col}_quarter'] = df[date_col].dt.quarter
        df[f'{date_col}_weekday'] = df[date_col].dt.dayofweek
        df[f'{date_col}_weekday_name'] = df[date_col].dt.day_name()
        df[f'{date_col}_is_weekend'] = df[date_col].dt.weekday >= 5
        df[f'{date_col}_day_of_year'] = df[date_col].dt.dayofyear
        df[f'{date_col}_week_of_year'] = df[date_col].dt.isocalendar().week
        
        # 创建季节特征
        df[f'{date_col}_season'] = df[date_col].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        logger.info(f"从 {date_col} 创建了日期特征")
        return df
        
    except Exception as e:
        logger.error(f"创建日期特征失败: {e}")
        return df


def handle_fra_missing_values(df: pd.DataFrame, 
                            high_missing_threshold: float = 0.95,
                            numeric_missing_threshold: float = 0.10,
                            text_missing_threshold: float = 0.20,
                            placeholder: str = 'UNKNOWN') -> pd.DataFrame:
    """
    智能处理FRA REAIR数据中的缺失值
    
    Args:
        df: 数据框
        high_missing_threshold: 高缺失值阈值，超过此比例的列将被删除
        numeric_missing_threshold: 数值型列缺失值阈值，低于此比例将进行填充
        text_missing_threshold: 文本型列缺失值阈值，低于此比例将进行填充
        placeholder: 文本型列缺失值填充占位符
        
    Returns:
        pd.DataFrame: 处理后的数据框
    """
    df = df.copy()
    original_shape = df.shape
    
    logger.info(f"开始处理缺失值，原始数据形状: {original_shape}")
    
    # 1. 计算每列的缺失值比例
    missing_ratio = df.isnull().sum() / len(df)
    
    # 2. 删除高缺失值列
    high_missing_cols = missing_ratio[missing_ratio > high_missing_threshold].index.tolist()
    if high_missing_cols:
        logger.info(f"删除高缺失值列 ({len(high_missing_cols)}个): {high_missing_cols}")
        df = df.drop(columns=high_missing_cols)
    
    # 3. 分别处理数值型和文本型列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    text_cols = df.select_dtypes(include=['object']).columns
    
    logger.info(f"数值型列: {len(numeric_cols)}个, 文本型列: {len(text_cols)}个")
    
    # 4. 处理数值型列
    numeric_filled = 0
    for col in numeric_cols:
        missing_ratio_col = df[col].isnull().sum() / len(df)
        if missing_ratio_col > 0 and missing_ratio_col <= numeric_missing_threshold:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            numeric_filled += 1
            logger.info(f"数值列 {col} 用中位数填充: {median_value}")
    
    # 5. 处理文本型列
    text_filled = 0
    for col in text_cols:
        missing_ratio_col = df[col].isnull().sum() / len(df)
        if missing_ratio_col > 0 and missing_ratio_col <= text_missing_threshold:
            df[col] = df[col].fillna(placeholder)
            text_filled += 1
            logger.info(f"文本列 {col} 用占位符填充: {placeholder}")
    
    # 6. 生成处理报告
    final_missing_ratio = df.isnull().sum() / len(df)
    remaining_high_missing = final_missing_ratio[final_missing_ratio > 0.5].index.tolist()
    
    logger.info(f"缺失值处理完成:")
    logger.info(f"  删除列数: {len(high_missing_cols)}")
    logger.info(f"  数值列填充数: {numeric_filled}")
    logger.info(f"  文本列填充数: {text_filled}")
    logger.info(f"  剩余高缺失值列: {len(remaining_high_missing)}")
    logger.info(f"  最终数据形状: {df.shape}")
    
    return df


def suggest_missing_value_strategies(df: pd.DataFrame) -> Dict[str, Any]:
    """
    为FRA数据提供缺失值处理策略建议
    
    Args:
        df: 数据框
        
    Returns:
        Dict: 处理策略建议
    """
    strategies = {
        'high_missing_columns': [],
        'numeric_columns': [],
        'text_columns': [],
        'special_columns': [],
        'recommendations': []
    }
    
    missing_ratio = df.isnull().sum() / len(df)
    
    # 1. 高缺失值列（建议删除）
    high_missing = missing_ratio[missing_ratio > 0.95].index.tolist()
    strategies['high_missing_columns'] = high_missing
    
    # 2. 数值型列处理建议
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        ratio = missing_ratio[col]
        if 0 < ratio <= 0.10:
            strategies['numeric_columns'].append({
                'column': col,
                'missing_ratio': ratio,
                'strategy': 'median_imputation',
                'description': f'用中位数填充 (缺失率: {ratio:.2%})'
            })
        elif 0.10 < ratio <= 0.50:
            strategies['numeric_columns'].append({
                'column': col,
                'missing_ratio': ratio,
                'strategy': 'advanced_imputation',
                'description': f'使用高级填充方法 (缺失率: {ratio:.2%})'
            })
    
    # 3. 文本型列处理建议
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        ratio = missing_ratio[col]
        if 0 < ratio <= 0.20:
            strategies['text_columns'].append({
                'column': col,
                'missing_ratio': ratio,
                'strategy': 'placeholder',
                'description': f'用占位符填充 (缺失率: {ratio:.2%})'
            })
        elif 0.20 < ratio <= 0.50:
            strategies['text_columns'].append({
                'column': col,
                'missing_ratio': ratio,
                'strategy': 'mode_imputation',
                'description': f'用众数填充 (缺失率: {ratio:.2%})'
            })
    
    # 4. 特殊列处理建议
    special_patterns = {
        'Railroad': ['Railroad Code', 'Railroad Name'],
        'Location': ['State', 'County', 'City'],
        'Accident': ['Accident Type', 'Cause'],
        'Equipment': ['Equipment Type', 'Car Number'],
        'Time': ['Date', 'Time', 'Year', 'Month']
    }
    
    for category, patterns in special_patterns.items():
        for pattern in patterns:
            matching_cols = [col for col in df.columns if pattern.lower() in col.lower()]
            for col in matching_cols:
                if col in missing_ratio.index and missing_ratio[col] > 0:
                    strategies['special_columns'].append({
                        'column': col,
                        'category': category,
                        'missing_ratio': missing_ratio[col],
                        'suggested_strategy': f'{category}特定处理'
                    })
    
    # 5. 总体建议
    strategies['recommendations'] = [
        "删除缺失率>95%的列，这些列信息价值很低",
        "对缺失率<10%的数值列使用中位数填充",
        "对缺失率<20%的文本列使用'UNKNOWN'占位符",
        "对铁路公司相关列，考虑用'Unknown Railroad'填充",
        "对地理位置列，考虑用'Unknown Location'填充",
        "对事故类型列，考虑用'Unknown Type'填充",
        "对时间相关列，考虑用前向填充或后向填充",
        "对设备相关列，考虑用'Unknown Equipment'填充"
    ]
    
    return strategies


def advanced_missing_value_imputation(df: pd.DataFrame, 
                                    column: str, 
                                    method: str = 'knn') -> pd.DataFrame:
    """
    高级缺失值填充方法
    
    Args:
        df: 数据框
        column: 要填充的列名
        method: 填充方法 ('knn', 'forward_fill', 'backward_fill', 'interpolation')
        
    Returns:
        pd.DataFrame: 填充后的数据框
    """
    df = df.copy()
    
    if column not in df.columns:
        logger.warning(f"列 {column} 不存在")
        return df
    
    if df[column].isnull().sum() == 0:
        logger.info(f"列 {column} 无缺失值")
        return df
    
    try:
        if method == 'knn':
            # KNN填充（需要scikit-learn）
            try:
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(n_neighbors=5)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    df_temp = df[numeric_cols].copy()
                    df_temp_imputed = pd.DataFrame(
                        imputer.fit_transform(df_temp),
                        columns=numeric_cols,
                        index=df.index
                    )
                    df[column] = df_temp_imputed[column]
                    logger.info(f"使用KNN填充列 {column}")
            except ImportError:
                logger.warning("scikit-learn未安装，使用中位数填充")
                df[column] = df[column].fillna(df[column].median())
        
        elif method == 'forward_fill':
            df[column] = df[column].fillna(method='ffill')
            logger.info(f"使用前向填充列 {column}")
        
        elif method == 'backward_fill':
            df[column] = df[column].fillna(method='bfill')
            logger.info(f"使用后向填充列 {column}")
        
        elif method == 'interpolation':
            if df[column].dtype in [np.number]:
                df[column] = df[column].interpolate()
                logger.info(f"使用插值填充列 {column}")
            else:
                logger.warning(f"列 {column} 不是数值型，无法使用插值")
        
        else:
            logger.warning(f"未知的填充方法: {method}")
        
        return df
        
    except Exception as e:
        logger.error(f"高级填充失败: {e}")
        return df
