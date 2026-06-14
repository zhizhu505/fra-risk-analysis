import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import spearmanr

# ================= 配置区 =================
# 结果根目录
RESULTS_ROOT = os.path.join('results', 'reports')
# 所有的州
STATES = ['California', 'Illinois', 'Texas']
# 模型版本文件夹名称
MODEL_VERSION = "xgb_baseline_v1"

# 所有的权重配置
# 格式: (w1, w2) -> 对应子文件夹 fusion_w1_{w1}_w2_{w2}
WEIGHT_CONFIGS = [
    (0.1, 0.9),
    (0.3, 0.7),
    (0.5, 0.5), # 基准 (Baseline)
    (0.7, 0.3),
    (0.9, 0.1)
]
BASELINE_CONFIG = (0.5, 0.5)

# 设置绘图风格
sns.set(style="whitegrid", font_scale=1.1)
# 尝试使用支持中文或通用字体，避免乱码
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ================= 数据加载函数 =================

def load_data_for_state(state):
    """
    读取指定州的所有权重配置下的 CSV 文件
    """
    data_map = {} # Key: (w1, w2), Value: DataFrame
    
    for w1, w2 in WEIGHT_CONFIGS:
        # 1. 构建文件夹路径
        # 例如: results/reports/California/xgb_baseline_v1/fusion_w1_0.1_w2_0.9
        folder_name = f"fusion_w1_{w1}_w2_{w2}"
        folder_path = os.path.join(RESULTS_ROOT, state, MODEL_VERSION, folder_name)
        
        # 2. 构建文件名
        # 根据你的截图，文件名格式为: {state}_xgb_baseline_v1_class_a_risks.csv
        # 注意：文件名中的 state 是小写的 (例如 california)
        filename = f"{state.lower()}_{MODEL_VERSION}_class_a_risks.csv"
        file_path = os.path.join(folder_path, filename)
        
        # 3. 读取文件
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                
                # 检查是否包含必要的列
                if 'CRI_Score' in df.columns and 'accident_type' in df.columns and 'evidence' in df.columns:
                    # 确保按 CRI 降序排列 (Rank 1 在最前)
                    df = df.sort_values('CRI_Score', ascending=False).reset_index(drop=True)
                    
                    # 添加排名列 (1-based)
                    df['Rank'] = df.index + 1
                    
                    # 创建唯一标识符 (Accident Type + Evidence 组合)
                    # 这样我们可以跨文件追踪同一个风险场景
                    df['ID'] = df['accident_type'].astype(str) + " | " + df['evidence'].astype(str)
                    
                    data_map[(w1, w2)] = df
                else:
                    print(f"  [警告] 文件 {filename} 缺少必要列 (CRI_Score/accident_type/evidence)，跳过。")
            except Exception as e:
                print(f"  [错误] 无法读取文件 {file_path}: {e}")
        else:
            print(f"  [警告] 文件不存在: {file_path}")
            
    return data_map

# ================= 分析与绘图函数 =================

def calculate_stability_metrics(data_map, state_name):
    """
    计算相对于 Baseline (0.5, 0.5) 的相关性和重合率
    """
    if BASELINE_CONFIG not in data_map:
        print(f"  [跳过] {state_name} 缺少基准配置 (0.5, 0.5) 的数据")
        return []

    df_base = data_map[BASELINE_CONFIG]
    
    # 获取基准 Top 20 的 ID 集合 (用于计算重合率)
    # 注意：如果文件行数不足20，就取全部
    top_n = 20
    base_top20_ids = set(df_base.head(top_n)['ID'])
    
    metrics = []
    
    for config, df_curr in data_map.items():
        w1, w2 = config
        
        # --- 指标 1: Spearman 排名相关系数 (Rank Correlation) ---
        # 我们只计算两个列表中都存在的风险场景的相关性
        common_ids = list(set(df_base['ID']).intersection(set(df_curr['ID'])))
        
        if len(common_ids) > 5: # 只有当共有元素足够多时计算才有意义
            # 提取这些共有元素在两个列表中的排名
            rank_base = df_base.set_index('ID').loc[common_ids]['Rank']
            rank_curr = df_curr.set_index('ID').loc[common_ids]['Rank']
            corr, _ = spearmanr(rank_base, rank_curr)
        else:
            corr = 0 # 或者 np.nan
            
        # --- 指标 2: Top 20 重合率 (Overlap Ratio) ---
        curr_top20_ids = set(df_curr.head(top_n)['ID'])
        if len(base_top20_ids) > 0:
            overlap_count = len(base_top20_ids.intersection(curr_top20_ids))
            overlap_ratio = overlap_count / len(base_top20_ids)
        else:
            overlap_ratio = 0
        
        metrics.append({
            'State': state_name,
            'Rule_Weight': w1,
            'Model_Weight': w2,
            'Spearman_Corr': corr,
            'Top20_Overlap': overlap_ratio
        })
        
    return pd.DataFrame(metrics)

def plot_bump_chart(data_map, state_name):
    """
    绘制排名变化图 (Bump Chart) - 展示 Baseline 前 5 名的变化
    """
    if BASELINE_CONFIG not in data_map:
        return

    df_base = data_map[BASELINE_CONFIG]
    # 取出基准排名前5的风险 ID
    top_5_df = df_base.head(5)
    top_5_ids = top_5_df['ID'].tolist()
    top_5_types = top_5_df['accident_type'].tolist()
    
    plot_data = []
    
    # 遍历所有权重配置
    # 先对权重配置排序，保证 X 轴顺序正确 (0.1 -> 0.9)
    sorted_configs = sorted(data_map.keys(), key=lambda x: x[0])
    
    for config in sorted_configs:
        w1, _ = config
        df_curr = data_map[config]
        
        for idx, target_id in enumerate(top_5_ids):
            # 查找该风险在当前配置中的排名
            record = df_curr[df_curr['ID'] == target_id]
            if not record.empty:
                rank = record.iloc[0]['Rank']
            else:
                # 如果跌出了榜单 (A类列表)，为了绘图可以设为一个较大的数，或者不画
                # 这里设为 DataFrame 长度 + 5，表示跌出视野
                rank = len(df_curr) + 5 
            
            # 为了图例简洁，只显示事故类型
            risk_label = f"#{idx+1}: {top_5_types[idx]}"
            
            plot_data.append({
                'Weight (Rule)': w1,
                'Rank': rank,
                'Risk Scenario': risk_label
            })
    
    if not plot_data:
        return

    df_plot = pd.DataFrame(plot_data)
    
    # 开始绘图
    plt.figure(figsize=(10, 6))
    
    # 绘制折线
    sns.lineplot(data=df_plot, x='Weight (Rule)', y='Rank', hue='Risk Scenario', 
                 marker='o', markersize=8, linewidth=2.5, palette='tab10')
    
    # 设置 Y 轴反转 (排名 1 在最上面)
    plt.gca().invert_yaxis()
    
    # 设置坐标轴标签
    plt.title(f'Rank Stability of Top 5 Risks in {state_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Rank Position', fontsize=12)
    plt.xlabel('Weight of Rule Path ($\\alpha$)', fontsize=12)
    
    # 设置 X 轴刻度
    plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    
    # 设置 Y 轴刻度 (显示整数排名)
    max_rank = df_plot[df_plot['Rank'] < 100]['Rank'].max() # 忽略跌出榜单的极端值
    if pd.isna(max_rank): max_rank = 10
    plt.yticks(range(1, int(max_rank) + 2))
    
    # 调整图例位置
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(RESULTS_ROOT, f'{state_name}_sensitivity_bump_chart.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  -> Bump Chart 已保存: {save_path}")
    plt.close() # 关闭图形，释放内存

# ================= 主程序 =================

if __name__ == "__main__":
    print(f"开始分析... 结果路径: {RESULTS_ROOT}")
    
    all_metrics_list = []

    for state in STATES:
        print(f"\n--- 处理州: {state} ---")
        
        # 1. 加载数据
        data_map = load_data_for_state(state)
        
        if not data_map:
            print(f"  [提示] 未加载到 {state} 的数据，跳过。")
            continue
            
        # 2. 计算指标 (相关性 & 重合率)
        df_metrics = calculate_stability_metrics(data_map, state)
        if not df_metrics.empty:
            all_metrics_list.append(df_metrics)
            # 打印简报
            print("  稳定性指标预览:")
            print(df_metrics[['Rule_Weight', 'Spearman_Corr', 'Top20_Overlap']].to_string(index=False))
        
        # 3. 绘制排名流向图 (Bump Chart)
        plot_bump_chart(data_map, state)

    # --- 汇总分析：绘制热力图 ---
    if all_metrics_list:
        df_all = pd.concat(all_metrics_list)
        
        # 保存所有指标到 CSV
        csv_path = os.path.join(RESULTS_ROOT, 'sensitivity_analysis_metrics.csv')
        df_all.to_csv(csv_path, index=False)
        print(f"\n-> 所有指标已保存至: {csv_path}")
        
        # 绘制热力图 (展示不同州的 Spearman 相关性)
        # 行: 州, 列: 权重配置
        pivot_corr = df_all.pivot(index='State', columns='Rule_Weight', values='Spearman_Corr')
        
        plt.figure(figsize=(8, 4))
        sns.heatmap(pivot_corr, annot=True, fmt=".3f", cmap="Blues", vmin=0.85, vmax=1.0,
                   linewidths=1, linecolor='white',
                   cbar_kws={'label': 'Spearman Rank Correlation'})
        
        plt.title('Stability Heatmap: Rank Correlation vs. Rule Weight', fontsize=12, fontweight='bold')
        plt.xlabel('Weight of Rule Path ($\\alpha$)', fontsize=11)
        plt.ylabel('')
        plt.tight_layout()
        
        heatmap_path = os.path.join(RESULTS_ROOT, 'sensitivity_stability_heatmap.png')
        plt.savefig(heatmap_path, dpi=300)
        print(f"-> 稳定性热力图已保存: {heatmap_path}")
        plt.show()
    else:
        print("\n[结束] 没有生成任何指标数据，请检查文件路径和内容格式。")
