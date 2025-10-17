import os
import argparse
from datetime import datetime
import pandas as pd
try:
	from src.text_processor import compute_text_column_quality, save_length_histogram, configure_matplotlib_chinese
except ModuleNotFoundError:
	from text_processor import compute_text_column_quality, save_length_histogram, configure_matplotlib_chinese


def main():
	parser = argparse.ArgumentParser(description="Generate quality report for a text column.")
	parser.add_argument("--csv", required=True, help="Path to CSV file")
	parser.add_argument("--column", default="text_merged", help="Text column name (default: text_merged)")
	parser.add_argument("--bins", type=int, default=50, help="Histogram bins (default: 50)")
	args = parser.parse_args()

	df = pd.read_csv(args.csv, low_memory=False)
	if args.column not in df.columns:
		raise ValueError(f"Column '{args.column}' not found in {args.csv}")

	series = df[args.column]
	metrics = compute_text_column_quality(series)

	print("缺失率（NaN或空字符串）：{:.4%}".format(metrics["missing_rate"]))
	lens = metrics["lengths"]
	print("字符长度统计（非空样本）：")
	print(f"- 最小: {lens['min']}")
	print(f"- 最大: {lens['max']}")
	print(f"- 平均: {lens['mean']:.2f}")
	print(f"- 标准差: {lens['std']:.2f}")

	if metrics["top_tokens"]:
		print("最常出现的 10 个词：")
		for w, c in metrics["top_tokens"]:
			print(f"- {w}: {c}")
	else:
		print("无可用词元，跳过Top-10词统计。")

	sample_n = len(metrics["samples"])
	if sample_n:
		print("\ntext_merged 随机抽样（5条或不足5条）：")
		for s in metrics["samples"]:
			print("-", s)
	else:
		print("\n无非空样本可抽样。")

	# 配置中文字体，保存直方图
	configure_matplotlib_chinese()
	viz_dir = os.path.join("results", "visualizations")
	os.makedirs(viz_dir, exist_ok=True)
	png_path = os.path.join(viz_dir, f"{args.column}_length_hist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
	save_length_histogram(series, png_path, bins=args.bins)
	print(f"直方图已保存: {png_path}")


if __name__ == "__main__":
	main()
