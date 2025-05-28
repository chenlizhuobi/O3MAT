import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 设置字体为新罗马
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取气候区域映射文件
region_map = pd.read_csv('/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/2011_299*459_9ClimateRegions.csv')
region_map = region_map[['ROW', 'COL', 'ClimateRegion']].dropna()
region_map['ClimateRegion'] = region_map['ClimateRegion'].astype(int)
region_map = region_map[region_map['ClimateRegion'].between(0, 8)]  # 仅保留0-8的气候区域

# 定义NOAA气候区域映射 (RegionCode -> CMAQName)
region_mapping = {
    '001': {'id': 0, 'name': 'Northeast', 'cmaq_name': 'NE_CR'},
    '002': {'id': 1, 'name': 'Northern Rockies and Plains', 'cmaq_name': 'NRP_CR'},
    '003': {'id': 2, 'name': 'Northwest', 'cmaq_name': 'NW_CR'},
    '004': {'id': 3, 'name': 'Ohio Valley', 'cmaq_name': 'CEN_CR'},
    '005': {'id': 4, 'name': 'South', 'cmaq_name': 'S_CR'},
    '006': {'id': 5, 'name': 'Southeast', 'cmaq_name': 'SE_CR'},
    '007': {'id': 6, 'name': 'Southwest', 'cmaq_name': 'SW_CR'},
    '008': {'id': 7, 'name': 'Upper Midwest', 'cmaq_name': 'UPMW_CR'},
    '009': {'id': 8, 'name': 'West', 'cmaq_name': 'W_CR'},
    'USA': {'id': 9, 'name': 'USA', 'cmaq_name': 'USA'}
}

# 定义需要绘图的指标
metrics = ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone', 'ds_ozone', 'harvard_ml']

# 变量名映射（替换为更友好的显示名称）
method_display_names = {
    'model': 'Model',
    'vna_ozone': 'VNA',
    'evna_ozone': 'eVNA',
    'avna_ozone': 'aVNA',
    'ds_ozone': 'Downscaler',
    'harvard_ml': 'Harvard ML',
}

# 定义Periods（可自定义）
periods = ['W126']  # 示例：只处理top-10时段

# 定义不同方法的标记样式
method_markers = {
    'model': '*'  ,
    'vna_ozone': 'o',     # 圆形
    'evna_ozone': 's',    # 正方形
    'avna_ozone': '^',    # 上三角形
    'ds_ozone': 'D',      # 菱形
    'harvard_ml': 'v',    # 下三角形
}

# 自定义Y轴范围（如果为None则自动计算）
y_limits = (0,28)  #  # 针对所有变量统一设置Y轴范围,对于top-10的变量，Y轴范围为(49, 96),其余指标(22.7,65),W126为(0,28)

# 读取原始数据
years = list(range(2002, 2020))
all_data = pd.DataFrame()

print("正在读取每年的数据...")
for year in tqdm(years):
    #FtA文件
    file_path = f'/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/{year}_W126_ST_Limit.csv'

    #AtF文件
    file_path = f'/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/{year}_W126_AtF.csv'
    try:
        data = pd.read_csv(file_path)
        data['Year'] = year
        # 合并气候区域信息
        data = data.merge(region_map, on=['ROW', 'COL'], how='left')
        all_data = pd.concat([all_data, data], ignore_index=True)
    except FileNotFoundError:
        print(f"未找到 {file_path} 文件。")

# 创建输出目录
output_dir = '/DeepLearning/mnt/shixiansheng/data_fusion/output/9ClimateRegion_Timeseries'
os.makedirs(output_dir, exist_ok=True)

# 为每个气候区、指标和Period生成时间序列图
print("正在生成时间序列图...")
for period in tqdm(periods, desc="处理时间段"):
    # 按Period筛选数据
    period_data = all_data[all_data['Period'] == period].copy()
    
    if period_data.empty:
        print(f"警告: 时间段 {period} 没有数据，跳过绘图")
        continue
    
    for region_code, region_info in tqdm(region_mapping.items(), desc="处理区域", leave=False):
        region_id = region_info['id']
        region_name = region_info['name']
        region_cmaq_name = region_info['cmaq_name']
        
        # 创建区域数据
        if region_id == 9:  # 全美国
            region_period_data = period_data.copy()
        else:
            region_period_data = period_data[period_data['ClimateRegion'] == region_id].copy()
        
        if region_period_data.empty:
            print(f"警告: 区域 {region_name} ({region_cmaq_name}) 在时间段 {period} 没有数据，跳过绘图")
            continue
        
        # 绘制一张图，包含所有方法的时间序列
        plt.figure(figsize=(12, 8))
        
        # 绘制各方法的时间序列
        methods = ['model','vna_ozone', 'evna_ozone', 'avna_ozone', 'ds_ozone', 'harvard_ml']
        colors = ['#8c564b','#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        min_val = float('inf')
        max_val = float('-inf')
        
        for i, method in enumerate(methods):
            if method in region_period_data.columns:
                # 特殊处理 harvard_ml：仅使用2002-2016年的数据
                if method == 'harvard_ml':
                    method_data = region_period_data[region_period_data['Year'] <= 2016].copy()
                else:
                    method_data = region_period_data.copy()
                
                # 获取显示名称和标记样式
                display_name = method_display_names.get(method, method)
                marker = method_markers.get(method, 'o')
                
                # 按年份聚合数据（对每个气候区的所有网格求平均作为浓度）
                yearly_data = method_data.groupby('Year')[method].mean().reset_index()
                
                # 检查是否有数据
                if not yearly_data.empty:
                    plt.plot(yearly_data['Year'], yearly_data[method], marker=marker, markersize=8, 
                             linestyle='-', linewidth=2, color=colors[i], label=display_name)
                    
                    # 更新Y轴范围计算
                    method_min = yearly_data[method].min()
                    method_max = yearly_data[method].max()
                    min_val = min(min_val, method_min)
                    max_val = max(max_val, method_max)
        
        # 设置Y轴范围（如果自定义）
        if y_limits is not None:
            plt.ylim(y_limits)
        else:
            # 预留一定空间
            if min_val != float('inf') and max_val != float('-inf'):
                padding = (max_val - min_val) * 0.1
                plt.ylim(min_val - padding, max_val + padding)
        
        # 设置图表标题和轴标签
        plt.title(f'{period}: {region_name} O₃ Time Series (2002-2019)', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('O₃ (ppm-hrs)', fontsize=14)  # 移除Concentration，直接使用O₃ (ppbv)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(years, rotation=45, fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        
        # 保存图表
        filename = f"{output_dir}/{period}_{region_cmaq_name}_Ozone_Timeseries.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
print("所有时间序列图生成完成！")    