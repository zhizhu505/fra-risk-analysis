import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast  # 用于解析字符串形式的字典

# ================= 配置区 =================
# CSV 文件所在的目录
RESULTS_DIR = os.path.join('results', 'reports')

# 文件名映射 (State Code -> Filename)
FILES = {
    'California (CA)': 'california_final_pareto_risk_scenarios.csv',
    'Illinois (IL)': 'illinois_final_pareto_risk_scenarios.csv',
    'Texas (TX)': 'texas_final_pareto_risk_scenarios.csv'
}

# 设置绘图风格 (学术风)
sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']  # 优先使用 Arial
plt.rcParams['axes.unicode_minus'] = False

# ================= 数据加载与处理函数 =================

def load_and_process_data(results_dir, file_map):
    """
    加载所有州的 A 类风险数据，并提取 Top 20 用于绘图
    """
    df_list = []
    feature_stats = []

    for state_name, filename in file_map.items():
        file_path = os.path.join(results_dir, filename)
        
        try:
            # 读取 CSV
            df = pd.read_csv(file_path)
            
            # 过滤 A 类风险 (如果文件中还没过滤)
            if 'Class' in df.columns:
                df = df[df['Class'] == 'A']
            
            # 取 Top 20 (按 CRI_Score 降序)
            # 如果没有 CRI_Score，假设数据已经是排序好的
            if 'CRI_Score' in df.columns:
                df = df.sort_values('CRI_Score', ascending=False)
            
            df_top20 = df.head(20).copy()
            df_top20['State'] = state_name
            
            # --- 1. 收集风险类型数据 ---
            df_list.append(df_top20[['State', 'accident_type']])
            
            # --- 2. 收集特征重要性数据 ---
            # 解析 matched_details 列提取特征名
            # 假设格式如: "Track Type=Main(Score:0.0113); Passengers..."
            for _, row in df_top20.iterrows():
                details = str(row['matched_details'])
                # 简单的字符串处理提取特征名
                # 分割分号 -> 分割括号 -> 分割等号
                items = details.split(';')
                for item in items:
                    if '(' in item:
                        # 提取特征名 (例如 "Track Type")
                        feature_raw = item.split('(')[0].strip()
                        if '=' in feature_raw:
                            feature_name = feature_raw.split('=')[0].strip()
                        else:
                            feature_name = feature_raw
                        
                        # 提取分数 (例如 "Score:0.0113")
                        try:
                            score_str = item.split('Score:')[1].split(')')[0]
                            score = float(score_str)
                            
                            feature_stats.append({
                                'State': state_name,
                                'Feature': feature_name,
                                'Importance': score
                            })
                        except (IndexError, ValueError):
                            continue # 解析失败跳过

        except FileNotFoundError:
            print(f"警告: 找不到文件 {file_path}，跳过该州。")

    # 合并数据
    df_risks = pd.concat(df_list, ignore_index=True)
    df_features = pd.DataFrame(feature_stats)
    
    return df_risks, df_features

# ================= 绘图函数 =================

def plot_stacked_bar(df_risks):
    """
    绘制图 1: 风险类型分布堆叠柱状图
    """
    # 统计每个州每种事故类型的数量
    risk_counts = df_risks.groupby(['State', 'accident_type']).size().unstack(fill_value=0)
    
    # 归一化为百分比 (可选，或者直接用数量)
    # risk_props = risk_counts.div(risk_counts.sum(axis=1), axis=0) * 100 
    
    plt.figure(figsize=(10, 6))
    
    # 绘图     # 使用 Viridis 调色板，区分度高且学术
    risk_counts.plot(kind='bar', stacked=True, colormap='viridis', width=0.65, ax=plt.gca())
    
    plt.title('Distribution of Top 20 Class A Risk Scenarios by State', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Number of Scenarios (Count)', fontsize=12)
    plt.xlabel('')
    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(title='Accident Type', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(RESULTS_DIR, 'cross_state_risk_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {save_path}")
    plt.show()

def plot_feature_heatmap(df_features):
    """
    绘制图 2: 特征重要性热力图
    """
    # 聚合数据：计算每个特征在每个州的平均重要性
    # 也可以用 sum，取决于你想展示“强度”还是“频率”
    heatmap_data = df_features.groupby(['State', 'Feature'])['Importance'].mean().unstack(fill_value=0)
    
    # 筛选 Top N 最重要的特征 (避免图表过大)
    # 按所有州的平均重要性排序，取前 10 个
    top_features = heatmap_data.mean().sort_values(ascending=False).head(10).index
    heatmap_data_filtered = heatmap_data[top_features].T # 转置：行是特征，列是州
    
    plt.figure(figsize=(8, 7))
    
    # 绘图
    sns.heatmap(heatmap_data_filtered, 
                annot=True,       # 显示数值
                fmt=".4f",        # 保留4位小数
                cmap="Blues",     # 蓝色系，颜色越深越重要
                linewidths=1,     # 网格线宽度
                linecolor='white',
                cbar_kws={'label': 'Avg. Feature Importance (Gain)'})
    
    plt.title('Feature Importance Intensity Across States\n(Top 10 Drivers)', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Risk Factor', fontsize=12)
    plt.xlabel('')
    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(RESULTS_DIR, 'cross_state_feature_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {save_path}")
    plt.show()

# ================= 主程序 =================

if __name__ == "__main__":
    # 1. 加载数据
    print("正在加载数据...")
    df_risks, df_features = load_and_process_data(RESULTS_DIR, FILES)
    
    if not df_risks.empty:
        # 2. 绘制堆叠柱状图
        print("正在绘制风险分布图...")
        plot_stacked_bar(df_risks)
        
        # 3. 绘制热力图
        if not df_features.empty:
            print("正在绘制特征热力图...")
            plot_feature_heatmap(df_features)
        else:
            print("警告: 未能提取到特征数据，无法绘制热力图。请检查 'matched_details' 列格式。")
    else:
        print("错误: 未加载到任何数据，请检查文件路径和内容。")