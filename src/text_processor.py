"""
文本处理(包括LLM集成)函数

本模块提供文本处理功能，包括LLM集成进行文本分析。
"""

import re
import html
import unicodedata
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> str:
    """
    文本预处理
    
    Args:
        text: 原始文本
        
    Returns:
        str: 预处理后的文本
    """
    if pd.isna(text):
        return ""
    
    # 转换为字符串
    text = str(text)
    
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 转换为小写
    text = text.lower()
    
    return text


def extract_keywords(text: str, keywords: List[str]) -> List[str]:
    """
    从文本中提取关键词
    
    Args:
        text: 输入文本
        keywords: 关键词列表
        
    Returns:
        List[str]: 找到的关键词
    """
    found_keywords = []
    text_lower = text.lower()
    
    for keyword in keywords:
        if keyword.lower() in text_lower:
            found_keywords.append(keyword)
    
    return found_keywords


def preprocess_text_advanced(
    text: str,
    *,
    to_lower: bool = True,
    remove_html: bool = True,
    normalize_unicode: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_control_chars: bool = True,
    keep_basic_punct: bool = True,
    collapse_whitespace: bool = True,
    strip_edges: bool = True,
    digits_to_placeholder: bool = False,
    placeholder_digit: str = "0",
) -> str:
    """
    通用的文本预处理函数，适用于在 LLM 结构化抽取前做清洗与标准化。

    参数说明（可根据需要开关）：
    - to_lower: 是否转换为小写
    - remove_html: 是否移除 HTML 标签并进行 HTML 实体反转义
    - normalize_unicode: 是否进行 Unicode 规范化 (NFKC)
    - remove_urls: 是否移除 URL
    - remove_emails: 是否移除邮箱
    - remove_control_chars: 是否移除控制字符（如\u0000-\u001F、\u007F）
    - keep_basic_punct: 是否保留常见标点（.,;:!?-_/()[]{}'"）且清理其余符号
    - collapse_whitespace: 是否将连续空白压缩为单个空格
    - strip_edges: 是否去除首尾空白
    - digits_to_placeholder: 是否将所有数字归一化为占位符（如 0）
    - placeholder_digit: 数字占位符字符
    """
    if pd.isna(text):
        return ""

    s = str(text)

    # 1) HTML 处理：先反转义实体，再删除标签
    if remove_html:
        s = html.unescape(s)
        s = re.sub(r"<[^>]+>", " ", s)

    # 2) 移除 URL / 邮箱
    if remove_urls:
        s = re.sub(r"https?://\S+|www\.\S+", " ", s, flags=re.IGNORECASE)
    if remove_emails:
        s = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", " ", s)

    # 3) Unicode 规范化
    if normalize_unicode:
        s = unicodedata.normalize("NFKC", s)

    # 4) 移除控制字符
    if remove_control_chars:
        s = "".join(ch for ch in s if (ch == "\t" or ch == "\n" or ch == "\r" or not unicodedata.category(ch).startswith("C")))

    # 5) 数字归一化（可选）
    if digits_to_placeholder:
        s = re.sub(r"\d", placeholder_digit, s)

    # 6) 符号/标点处理
    if keep_basic_punct:
        # 保留一组基础标点，移除其他符号（Unicode 类别为 P 的标点之外的符号可按需扩展）
        allowed = set(".,;:!?-_/()[]{}'\"")
        s = "".join(ch if (ch.isalnum() or ch.isspace() or ch in allowed) else " " for ch in s)
    else:
        # 仅保留字母数字与空白
        s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)

    # 7) 空白与大小写规范
    if strip_edges:
        s = s.strip()
    if collapse_whitespace:
        s = re.sub(r"\s+", " ", s)
    if to_lower:
        s = s.lower()

    return s


def analyze_text_sentiment(text: str) -> Dict[str, Any]:
    """
    分析文本情感（基础版本）
    
    Args:
        text: 输入文本
        
    Returns:
        Dict: 情感分析结果
    """
    # 简单的情感分析关键词
    positive_words = ['good', 'safe', 'improved', 'better', 'success']
    negative_words = ['bad', 'dangerous', 'failed', 'worse', 'accident']
    
    text_lower = text.lower()
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    sentiment = 'neutral'
    if positive_count > negative_count:
        sentiment = 'positive'
    elif negative_count > positive_count:
        sentiment = 'negative'
    
    return {
        'sentiment': sentiment,
        'positive_score': positive_count,
        'negative_score': negative_count
    }


def integrate_llm_analysis(text: str, llm_client=None) -> Dict[str, Any]:
    """
    LLM集成分析（占位符）
    
    Args:
        text: 输入文本
        llm_client: LLM客户端
        
    Returns:
        Dict: LLM分析结果
    """
    # 这里是LLM集成的占位符
    # 实际实现需要根据具体的LLM API进行配置
    
    logger.info("LLM分析功能待实现")
    
    return {
        'llm_analysis': '待实现',
        'confidence': 0.0,
        'extracted_entities': []
    }

<<<<<<< HEAD

def preprocess_text_columns(
    df: pd.DataFrame,
    columns: List[str],
    *,
    suffix: str = "_clean",
    cleaner=preprocess_text_advanced,
    cleaner_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    对指定列批量进行文本清洗，生成带后缀的新列并返回新的 DataFrame。

    参数:
    - df: 输入 DataFrame
    - columns: 需要清洗的列名列表
    - suffix: 新列后缀（默认 _clean）
    - cleaner: 文本清洗函数（默认 preprocess_text_advanced）
    - cleaner_kwargs: 传给清洗函数的可选参数字典
    """
    if cleaner_kwargs is None:
        cleaner_kwargs = {}

    df_out = df.copy()
    for col in columns:
        if col in df_out.columns:
            df_out[col + suffix] = df_out[col].apply(lambda x: cleaner(x, **cleaner_kwargs))
    return df_out


def merge_text_columns(
    df: pd.DataFrame,
    columns: List[str],
    *,
    new_column: str = "text_merged",
    separator: str = " [SEP] ",
    strip_values: bool = True,
) -> pd.DataFrame:
    """
    合并多个文本列到单一列，按顺序连接，跳过空值/空白。

    参数:
    - df: 输入 DataFrame
    - columns: 需要合并的列名列表
    - new_column: 合并后新列名
    - separator: 列内容之间的分隔符（默认 " [SEP] ")
    - strip_values: 是否对每个单元格做 strip 去除首尾空白
    """
    def _merge_row(row: pd.Series) -> str:
        parts: List[str] = []
        for col in columns:
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    s = str(val)
                    if strip_values:
                        s = s.strip()
                    if s:
                        parts.append(s)
        return separator.join(parts)

    df_out = df.copy()
    df_out[new_column] = df_out[columns].apply(_merge_row, axis=1)
    return df_out


def compute_text_column_quality(
    series: pd.Series,
    *,
    top_n: int = 10,
    sample_k: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    计算文本列的质量指标：缺失率、长度分布统计、Top-N词频与随机样本。

    Returns 字段：
    - missing_rate: float
    - lengths: Dict[min, max, mean, std]
    - top_tokens: List[(token, count)]
    - samples: List[str]
    - non_empty_count: int
    - total_count: int
    """
    import re as _re
    from collections import Counter as _Counter

    total_count = int(series.shape[0])
    is_na_or_empty = series.isna() | (series.astype(str).str.strip() == "")
    missing_rate = float(is_na_or_empty.mean())

    non_empty = series.loc[~is_na_or_empty].astype(str)
    non_empty_count = int(non_empty.shape[0])
    lengths_series = non_empty.str.len()

    if non_empty_count > 0:
        lengths = {
            "min": int(lengths_series.min()),
            "max": int(lengths_series.max()),
            "mean": float(lengths_series.mean()),
            "std": float(lengths_series.std(ddof=1)),
        }
        # tokenization: a-z0-9
        tokens: List[str] = []
        for t in non_empty:
            tokens.extend(_re.findall(r"[a-z0-9]+", t.lower()))
        top_tokens = _Counter(tokens).most_common(top_n) if tokens else []
        sample_n = min(sample_k, non_empty_count)
        samples = non_empty.sample(n=sample_n, random_state=random_state).tolist() if sample_n > 0 else []
    else:
        lengths = {"min": 0, "max": 0, "mean": 0.0, "std": 0.0}
        top_tokens = []
        samples = []

    return {
        "missing_rate": missing_rate,
        "lengths": lengths,
        "top_tokens": top_tokens,
        "samples": samples,
        "non_empty_count": non_empty_count,
        "total_count": total_count,
    }


def save_length_histogram(series: pd.Series, png_path: str, *, bins: int = 50) -> None:
    """
    保存文本长度直方图（仅针对非空文本）。
    """
    import matplotlib.pyplot as _plt
    import os as _os

    non_empty = series.dropna()
    non_empty = non_empty[non_empty.astype(str).str.strip() != ""].astype(str)
    lengths = non_empty.str.len()
    _os.makedirs(_os.path.dirname(png_path), exist_ok=True)
    _plt.figure(figsize=(8, 4))
    _plt.hist(lengths, bins=bins, color="#4C72B0", alpha=0.85, edgecolor="white")
    _plt.title("text_merged length distribution (non-empty)")
    _plt.xlabel("length")
    _plt.ylabel("count")
    _plt.tight_layout()
    _plt.savefig(png_path, dpi=120)
    _plt.close()


def configure_matplotlib_chinese(preferred_fonts: Optional[List[str]] = None) -> List[str]:
    """
    配置 matplotlib 使用支持中文的字体，避免中文字符显示为方块或警告。

    preferred_fonts: 优先候选字体列表；默认尝试常见字体。
    返回：最终设置的 sans-serif 字体候选列表。
    """
    import matplotlib as _mpl

    if preferred_fonts is None:
        preferred_fonts = [
            "Microsoft YaHei",  # Windows 常见
            "SimHei",           # 黑体
            "SimSun",           # 宋体
            "Noto Sans CJK SC", # Noto 中文
            "WenQuanYi Zen Hei",
        ]

    # 设置字体候选与负号
    _mpl.rcParams["font.sans-serif"] = preferred_fonts + _mpl.rcParams.get("font.sans-serif", [])
    _mpl.rcParams["axes.unicode_minus"] = False

    return _mpl.rcParams["font.sans-serif"]

=======
>>>>>>> main
