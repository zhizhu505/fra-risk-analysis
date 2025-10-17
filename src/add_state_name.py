import os
from datetime import datetime
import pandas as pd


def add_state_name_to_target(
	source_csv: str,
	target_csv: str,
	state_col: str = "State Name",
	possible_keys=None,
) -> str:
	"""
	将源数据集中的州名列补充到目标 CSV 中。

	优先按索引（行数一致）对齐；若行数不一致且存在公共键，则尝试基于键合并。

	返回最终写入的文件路径。
	"""
	if possible_keys is None:
		possible_keys = [
			["Date", "Time", "Accident Type"],
			["Date", "Accident Type"],
			["Time", "Accident Type"],
		]

	src = pd.read_csv(source_csv, low_memory=False)
	tgt = pd.read_csv(target_csv, low_memory=False)

	if state_col not in src.columns:
		raise ValueError(f"'{state_col}' 不在源数据集中: {source_csv}")

	# 情况1：行数一致，直接按索引对齐
	if len(src) == len(tgt):
		tgt[state_col] = src[state_col].values
		return _safe_save(target_csv, tgt)

	# 情况2：尝试用公共键做合并（左连接保留目标行）
	for keys in possible_keys:
		if all(k in src.columns for k in keys) and all(k in tgt.columns for k in keys):
			merged = tgt.merge(
				src[keys + [state_col]].drop_duplicates(),
				how="left",
				on=keys,
				validate="m:1",
			)
			return _safe_save(target_csv, merged)

	raise RuntimeError(
		"无法补充州名：行数不一致且未找到可用公共键，请提供唯一键用于匹配。"
	)


def _safe_save(path: str, df: pd.DataFrame) -> str:
	"""保存到指定路径，若被占用则写入带时间戳的新文件，返回最终路径。"""
	dirname = os.path.dirname(path)
	if dirname:
		os.makedirs(dirname, exist_ok=True)
	try:
		df.to_csv(path, index=False)
		print(f"已写入: {path}")
		return path
	except PermissionError:
		ts = datetime.now().strftime("%Y%m%d_%H%M%S")
		alt_path = os.path.join(dirname, f"{os.path.splitext(os.path.basename(path))[0]}_{ts}.csv")
		df.to_csv(alt_path, index=False)
		print(f"目标被占用，已改存为: {alt_path}")
		return alt_path


if __name__ == "__main__":
	# 修改为你的实际路径
	source_csv = r"D:\railway_risk_analysis\data\processed\fra_reair_data_cleaned.csv"
	target_csv = r"D:\railway_risk_analysis\data\processed\df_selected_features_cleaned_20251017_151910.csv"
	add_state_name_to_target(source_csv, target_csv)


