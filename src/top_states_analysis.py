import pandas as pd
import os
from datetime import datetime
from typing import List, Tuple
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
try:
	from src.text_processor import preprocess_text_columns, merge_text_columns
except ModuleNotFoundError:
	# 兼容直接执行本文件: python src/top_states_analysis.py
	from text_processor import preprocess_text_columns, merge_text_columns


def compute_top_states(csv_path: str, top_n: int = 10, include_na: bool = False) -> pd.DataFrame:
	"""
	读取给定 CSV 文件，基于列 "State Name" 统计事故数量，按降序返回前 N 个州。

	参数:
	- csv_path: CSV 文件的绝对路径或相对路径
	- top_n: 返回前 N 条记录
	- include_na: 是否将缺失值 (NaN) 作为一个类别进行统计

	返回:
	- 包含两列的 DataFrame: ["State Name", "accident_count"]，按事故数降序
	"""
	# 1) 读取数据
	df = pd.read_csv(csv_path, low_memory=False)

	# 2) 计算每个州的事故总数
	# 使用 value_counts 对 "State Name" 列进行频数统计
	state_counts = (
		df["State Name"]
		.value_counts(dropna=not include_na)  # include_na=False -> dropna=True
		.rename("accident_count")
		.reset_index()  # 兼容旧版本 pandas，无 names 参数
		.rename(columns={"index": "State Name"})
	)

	# 3) 按事故数量降序排列（value_counts 默认已降序，这里显式排序保证可读性与稳健性）
	state_counts_sorted = state_counts.sort_values("accident_count", ascending=False)

	# 4) 取前 N 个州
	return state_counts_sorted.head(top_n)


def get_accident_type_mapping(csv_path: str) -> pd.DataFrame:
	"""
	读取列 "Accident Type Code" 与 "Accident Type"，生成唯一映射表。

	返回:
	- DataFrame，包含两列：accident_type_code, accident_type，按编码升序排列
	"""
	df = pd.read_csv(csv_path, low_memory=False, usecols=["Accident Type Code", "Accident Type"])
	mapping = (
		df[["Accident Type Code", "Accident Type"]]
		.dropna()
		.drop_duplicates()
		.sort_values("Accident Type Code")
		.rename(columns={
			"Accident Type Code": "accident_type_code",
			"Accident Type": "accident_type",
		})
	)
	return mapping


def _normalize(name: str) -> str:
	"""将列名做规范化：去首尾空格、压缩中间空格、统一小写。"""
	return " ".join(str(name).strip().split()).lower()


def select_core_features(df_cleaned: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
	"""
	从 df_cleaned 中仅保留核心特征列，返回子 DataFrame 及未匹配的列名列表。

	核心列名（按用户给定）：
	- Date
	- Time
	- Accident Type
	- Temperature
	- Visibility
	- Weather Condition
	- Equipment Type
	- Equipment Attended
	- Train Speed
	- Recorded Estimated Speed
	- Signalization
	- Method of Operation
	- Adjunct Name 1
	- Positive Alcohol Tests
	- Positive Drug Tests
	- Primary Accident Cause
	- Contributing Accident Cause
	"""
	desired_cols_raw: List[str] = [
		"Date",
		"Time",
		"Accident Type",
		"Temperature",
		"Visibility",
		"Weather Condition",
		"Equipment Type",
		"Equipment Attended",
		"Train Speed",
		"Recorded Estimated Speed",
		"Signalization",
		"Method of Operation",
		"Adjunct Name 1",
		"Positive Alcohol Tests",
		"Positive Drug Tests",
		"Primary Accident Cause",
		"Contributing Accident Cause",
	]

	# 建立规范化名到实际列名的映射
	normalized_to_actual = {_normalize(c): c for c in df_cleaned.columns}

	matched_actual_cols: List[str] = []
	missing_cols: List[str] = []

	for col in desired_cols_raw:
		key = _normalize(col)
		if key in normalized_to_actual:
			matched_actual_cols.append(normalized_to_actual[key])
		else:
			missing_cols.append(col)

	df_selected_features = df_cleaned.loc[:, matched_actual_cols].copy()
	return df_selected_features, missing_cols


if __name__ == "__main__":
	# 可根据需要调整文件路径与参数
	csv_path = r"D:\railway_risk_analysis\data\processed\fra_reair_data_cleaned.csv"

	# 计算并打印前 10 个州
	top10_states = compute_top_states(csv_path=csv_path, top_n=10, include_na=False)
	print("各州事故数量（前10，按降序）：")
	print(top10_states.to_string(index=False))

	# 打印事故类型编码与名称映射（应为 13 类）
	print("\n事故类型编码映射（预计 13 类）：")
	mapping_df = get_accident_type_mapping(csv_path)
	print(mapping_df.to_string(index=False))
	print(f"\n类别总数：{mapping_df.shape[0]}")

	# 计算并打印各州内事故类型占比（行归一化）
	def compute_state_type_percentage_crosstab(csv_path: str, include_na: bool = False, decimals: int = 2) -> pd.DataFrame:
		"""
		生成一个交叉表：行=州 (State Name)，列=事故类型 (Accident Type Code)，
		值为各州内部每种事故类型发生次数占比（百分比）。

		参数:
		- csv_path: CSV 文件路径
		- include_na: 是否将缺失值视为单独类别计入百分比
		- decimals: 百分比小数位数

		返回:
		- 行内归一化后的百分比表（0-100），DataFrame
		"""
		df = pd.read_csv(csv_path, low_memory=False)
		state_series = df["State Name"]
		type_series = df["Accident Type Code"]

		if include_na:
			state_series = state_series.fillna("(Missing)")
			type_series = type_series.fillna("(Missing)")
			ct = pd.crosstab(state_series, type_series, normalize="index")
		else:
			# 默认丢弃缺失，normalize='index' 会对每行按有效类别求比例
			ct = pd.crosstab(state_series, type_series, dropna=True, normalize="index")

		# 转为百分比并四舍五入
		ct = (ct * 100).round(decimals)
		ct.index.name = "State Name"
		ct.columns.name = "Accident Type Code"
		return ct

	print("\n各州内事故类型占比（%），行合计约为 100：")
	percent_xtab = compute_state_type_percentage_crosstab(csv_path=csv_path, include_na=False, decimals=2)
	print(percent_xtab.to_string())

	# 基于 df_cleaned 选择核心特征并展示前 5 行与列名
	print("\n核心特征选择示例（来源于 df_cleaned）：")
	df_cleaned = pd.read_csv(csv_path, low_memory=False)
	df_selected_features, missing_cols = select_core_features(df_cleaned)
	print("df_selected_features 前 5 行：")
	print(df_selected_features.head(5).to_string(index=False))
	print("\ndf_selected_features 列名：")
	print(list(df_selected_features.columns))
	if missing_cols:
		print("\n以下列名未在 df_cleaned 中找到，请检查：")
		for c in missing_cols:
			print(f"- {c}")

	# 对文本列进行批量清洗并保存
	text_columns = [
		"Date",
		"Time",
		"Accident Type",
		"Temperature",
		"Visibility",
		"Weather Condition",
		"Equipment Type",
		"Equipment Attended",
		"Train Speed",
		"Recorded Estimated Speed",
		"Signalization",
		"Method of Operation",
		"Adjunct Name 1",
		"Positive Alcohol Tests",
		"Positive Drug Tests",
		"Primary Accident Cause",
		"Contributing Accident Cause",
	]

	df_selected_features_cleaned = preprocess_text_columns(
		df_selected_features,
		[col for col in text_columns if col in df_selected_features.columns],
		cleaner_kwargs={
			"digits_to_placeholder": False,
		}
	)

	# 确保目录存在
	output_dir = os.path.join("data", "processed")
	os.makedirs(output_dir, exist_ok=True)
	output_path = os.path.join(output_dir, "df_selected_features_cleaned.csv")
	try:
		df_selected_features_cleaned.to_csv(output_path, index=False)
		print(f"\n已保存清洗后的特征到: {output_path}")
	except PermissionError:
		ts = datetime.now().strftime("%Y%m%d_%H%M%S")
		alt_output_path = os.path.join(output_dir, f"df_selected_features_cleaned_{ts}.csv")
		df_selected_features_cleaned.to_csv(alt_output_path, index=False)
		print(f"\n原始文件被占用，已改存为: {alt_output_path}")

	# 合并文本列为 text_merged 并打印前 5 个非空示例
	merge_cols = [
		"Primary Accident Cause",
		"Contributing Accident Cause",
		"Accident Type",
	]
	df_with_merged = merge_text_columns(
		df_selected_features_cleaned,
		[col for col in merge_cols if col in df_selected_features_cleaned.columns],
		new_column="text_merged",
		separator=" [SEP] ",
	)
	print("\ntext_merged 前 5 个非空示例：")
	print(
		df_with_merged.loc[df_with_merged["text_merged"].str.strip().ne(""), "text_merged"]
		.head(5)
		.to_string(index=False)
	)

	# 可选：覆盖保存（带 text_merged）
	try:
		df_with_merged.to_csv(output_path, index=False)
		print(f"\n已更新保存带 text_merged 的文件: {output_path}")
	except PermissionError:
		ts = datetime.now().strftime("%Y%m%d_%H%M%S")
		alt_output_path2 = os.path.join(output_dir, f"df_selected_features_cleaned_{ts}.csv")
		df_with_merged.to_csv(alt_output_path2, index=False)
		print(f"\n原始文件被占用，已改存为: {alt_output_path2}")

	# 对 text_merged 执行质量检查
	print("\ntext_merged 质量检查：")
	is_na_or_empty = df_with_merged["text_merged"].isna() | (df_with_merged["text_merged"].astype(str).str.strip() == "")
	missing_rate = float(is_na_or_empty.mean())
	print(f"缺失率（NaN或空字符串）：{missing_rate:.4%}")

	non_empty = df_with_merged.loc[~is_na_or_empty, "text_merged"].astype(str)
	lengths = non_empty.str.len()
	if len(lengths) > 0:
		print("字符长度统计（非空样本）：")
		print(f"- 最小: {int(lengths.min())}")
		print(f"- 最大: {int(lengths.max())}")
		print(f"- 平均: {float(lengths.mean()):.2f}")
		print(f"- 标准差: {float(lengths.std(ddof=1)):.2f}")

		# 保存直方图
		viz_dir = os.path.join("results", "visualizations")
		os.makedirs(viz_dir, exist_ok=True)
		png_path = os.path.join(viz_dir, f"text_merged_length_hist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
		plt.figure(figsize=(8, 4))
		plt.hist(lengths, bins=50, color="#4C72B0", alpha=0.85, edgecolor="white")
		plt.title("text_merged 字符长度分布（非空）")
		plt.xlabel("字符长度")
		plt.ylabel("频数")
		plt.tight_layout()
		plt.savefig(png_path, dpi=120)
		plt.close()
		print(f"直方图已保存: {png_path}")
	else:
		print("无非空文本，跳过长度统计与直方图。")

	# Top-10 词频
	def _tokenize(text: str):
		return re.findall(r"[a-z0-9]+", str(text).lower())

	tokens = []
	for t in non_empty:
		tokens.extend(_tokenize(t))
	if len(tokens) > 0:
		top10 = Counter(tokens).most_common(10)
		print("最常出现的 10 个词：")
		for w, c in top10:
			print(f"- {w}: {c}")
	else:
		print("无可用词元，跳过Top-10词统计。")

	# 抽样预览
	sample_n = min(5, len(non_empty))
	if sample_n > 0:
		print("\ntext_merged 随机抽样（5条或不足5条）：")
		for s in non_empty.sample(n=sample_n, random_state=42).tolist():
			print("-", s)
	else:
		print("\n无非空样本可抽样。")
