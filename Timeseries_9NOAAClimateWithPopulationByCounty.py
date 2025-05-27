# 定义NOAA气候区域映射 (RegionCode -> CMAQName)
region_mapping = {
    'Northeast': 'NE_CR',
    'Northern Rockies and Plains': 'NRP_CR',
    'Northwest': 'NW_CR',
    'Ohio Valley': 'CEN_CR',
    'South': 'S_CR',
    'Southeast': 'SE_CR',
    'Southwest': 'SW_CR',
    'Upper Midwest': 'UPMW_CR',
    'West': 'W_CR',
    'USA': 'USA'
}

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 设置字体为新罗马
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取县区域映射文件
region_map = pd.read_csv('/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/2011_299*459_CountyRegions.csv')
region_map = region_map[['ROW', 'COL', 'StateName', 'CountyName']].dropna()
# 排除不在任何州的区域
region_map = region_map[region_map['StateName'] != "-999"]  

# 读取县人口数据
population_data = pd.read_csv('/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/UnitedStatesCensusCountyPopulation2010to2020.csv', encoding='ISO-8859-1')
# 提取2010年人口数据并重命名列
county_population = population_data[['STATE_NAME', 'COUNTY_NAME', 'POPULATION_2010']].copy()
county_population.columns = ['StateName', 'CountyName', 'Population']

# 定义气候区到州的映射
climate_region_mapping = {
    'Northeast': [
        'Connecticut', 'Delaware', 'Maine', 'Maryland', 
        'Massachusetts', 'New Hampshire', 'New Jersey', 
        'New York', 'Pennsylvania', 'Rhode Island', 'Vermont'
    ],
    'Upper Midwest': [
        'Iowa', 'Michigan', 'Minnesota', 'Wisconsin'
    ],
    'Ohio Valley': [
        'Illinois', 'Indiana', 'Kentucky', 'Missouri', 
        'Ohio', 'Tennessee', 'West Virginia'
    ],
    'Southeast': [
        'Alabama', 'Florida', 'Georgia', 'North Carolina', 
        'South Carolina', 'Virginia'
    ],
    'Northern Rockies and Plains': [
        'Montana', 'Nebraska', 'North Dakota', 'South Dakota', 
        'Wyoming'
    ],
    'South': [
        'Arkansas', 'Kansas', 'Louisiana', 'Mississippi', 
        'Oklahoma', 'Texas'
    ],
    'Southwest': [
        'Arizona', 'Colorado', 'New Mexico', 'Utah'
    ],
    'Northwest': [
        'Idaho', 'Oregon', 'Washington'
    ],
    'West': [
        'California', 'Nevada'
    ],
    'USA': list(county_population['StateName'].unique())  # USA包含所有州
}

# 创建州到气候区的反向映射
state_to_climate_region = {}
for climate_region, states in climate_region_mapping.items():
    for state in states:
        state_to_climate_region[state] = climate_region

# 为region_map添加气候区信息
region_map['ClimateRegion'] = region_map['StateName'].map(state_to_climate_region)
# 过滤掉不在映射中的州
region_map = region_map.dropna(subset=['ClimateRegion'])

# 添加CMAQ名称映射
region_map['CMAQName'] = region_map['ClimateRegion'].map(region_mapping)

# 合并县人口数据到区域映射表
region_map = region_map.merge(county_population, on=['StateName', 'CountyName'], how='left')

# 找出没有人口数据的县
missing_population_counties = region_map[region_map['Population'].isna()][['StateName', 'CountyName']].drop_duplicates()
if not missing_population_counties.empty:
    print("警告: 以下县没有找到人口数据，将不参与计算:")
    for _, row in missing_population_counties.iterrows():
        print(f"  - {row['CountyName']}, {row['StateName']}")

# 过滤掉没有人口数据的县
region_map = region_map.dropna(subset=['Population'])

# 定义需要绘图的指标
metrics = ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone', 'ds_ozone', 'harvard_ml']

# 变量名映射（替换为更友好的显示名称）
method_display_names = {
    'vna_ozone': 'VNA',
    'evna_ozone': 'eVNA',
    'avna_ozone': 'aVNA',
    'ds_ozone': 'Downscaler',
    'harvard_ml': 'Harvard ML',
    'model': 'EQUATES'
}

# 定义Periods（可自定义）
periods = ['DJF', 'MAM', 'JJA', 'SON', 'Annual', 'Apr-Sep']
periods = ['W126']

# 定义不同方法的标记样式
method_markers = {
    'model': '*',          # 星形
    'vna_ozone': 'o',     # 圆形
    'evna_ozone': 's',    # 正方形
    'avna_ozone': '^',    # 上三角形
    'ds_ozone': 'D',      # 菱形
    'harvard_ml': 'v'     # 下三角形
}

# 自定义Y轴范围（如果为None则自动计算）
y_limits = (0,28)  # 针对所有变量统一设置Y轴范围,对于top-10的变量，Y轴范围为(49, 96),其余指标(22.7,65),W126为(0,28)

# 读取原始数据
years = list(range(2002, 2020))
all_data = pd.DataFrame()

print("正在读取每年的数据...")
for year in tqdm(years):
    file_path = f'/DeepLearning/mnt/shixiansheng/data_fusion/output/DailyData_WithoutCV/{year}_Data_WithoutCV_Metrics.csv'
    #小时计算指标W126
    file_path = f'/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/{year}_W126_ST_Limit.csv'
    file_path = f'/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/{year}_W126_AtF.csv'
    try:
        data = pd.read_csv(file_path)
        data['Year'] = year
        # 合并县区域信息和气候区信息
        data = data.merge(region_map, on=['ROW', 'COL'], how='left')
        all_data = pd.concat([all_data, data], ignore_index=True)
    except FileNotFoundError:
        print(f"未找到 {file_path} 文件。")

# 创建输出目录
output_dir = '/DeepLearning/mnt/shixiansheng/data_fusion/output/9ClimateRegion_PopWeightedCountyBased_Timeseries'
os.makedirs(output_dir, exist_ok=True)

# 为每个气候区、指标和Period生成时间序列图
print("正在生成时间序列图...")
for period in tqdm(periods, desc="处理时间段"):
    # 按Period筛选数据
    period_data = all_data[all_data['Period'] == period].copy()
    
    if period_data.empty:
        print(f"警告: 时间段 {period} 没有数据，跳过绘图")
        continue
    
    # 包含USA的所有气候区
    all_regions = list(climate_region_mapping.keys())
    
    for climate_region in tqdm(all_regions, desc="处理气候区", leave=False):
        # 获取该气候区包含的州
        states_in_region = climate_region_mapping[climate_region]
        
        # 创建区域数据（包含该气候区下的所有州）
        region_period_data = period_data[period_data['StateName'].isin(states_in_region)].copy()
        
        if region_period_data.empty:
            print(f"警告: 气候区 {climate_region} 在时间段 {period} 没有数据，跳过绘图")
            continue
        
        # 绘制一张图，包含所有方法的时间序列
        plt.figure(figsize=(12, 8))
        
        # 绘制各方法的时间序列
        methods = ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone', 'ds_ozone', 'harvard_ml']
        colors = ['#8c564b', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
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
                
                # 计算人口加权的时间序列
                yearly_data = pd.DataFrame()
                
                for year in years:
                    year_data = method_data[method_data['Year'] == year].copy()
                    
                    if year_data.empty:
                        continue
                    
                    # 按县分组计算平均浓度
                    county_avg = year_data.groupby(['StateName', 'CountyName']).agg(
                        {method: 'mean', 'Population': 'first'}
                    ).reset_index()
                    
                    # 检查是否有数据
                    if not county_avg.empty and 'Population' in county_avg.columns and not county_avg['Population'].isna().all():
                        # 计算人口加权浓度
                        pop_weighted_concentration = (county_avg[method] * county_avg['Population']).sum() / county_avg['Population'].sum()
                        
                        # 记录该年的人口加权浓度
                        yearly_data = pd.concat([
                            yearly_data, 
                            pd.DataFrame({'Year': [year], method: [pop_weighted_concentration]})
                        ], ignore_index=True)
                
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
        plt.title(f'{period}: {climate_region} O₃ Population-Weighted Time Series (2002-2019)', fontsize=16)
        plt.title(f'AtF_{period}: {climate_region} O₃ Population-Weighted Time Series (2002-2019)', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Population-Weighted O₃ (ppbv)', fontsize=14)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(years, rotation=45, fontsize=12)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        
        # 保存图表
        cmaq_name = region_mapping.get(climate_region, climate_region)
        filename = f"{output_dir}/{period}_{cmaq_name}_Ozone_PopWeighted_Timeseries_AtF.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
print("所有基于县的人口加权时间序列图生成完成！")    