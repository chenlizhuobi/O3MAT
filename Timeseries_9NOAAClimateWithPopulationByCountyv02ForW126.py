import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 设置字体为新罗马
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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
y_limits = (0, 28)  #  # 针对所有变量统一设置Y轴范围,对于top-10的变量，Y轴范围为(49, 96),其余指标(22.7,65),W126为(0,28)

# 读取原始数据
years = list(range(2002, 2020))
all_data = pd.DataFrame()

print("正在读取每年的数据...")
for year in tqdm(years):
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
    period_data = all_data[all_data['Period'] == period].copy()
    
    if period_data.empty:
        print(f"警告: 时间段 {period} 没有数据，跳过绘图")
        continue
    
    all_regions = list(climate_region_mapping.keys())
    
    # ----------------------
    # 计算全局Y轴范围（关键修正）
    # ----------------------
    global_min_val = float('inf')
    global_max_val = float('-inf')
    
    # 收集该Period下所有气候区、所有方法的人口加权数据
    all_weighted_data = []
    for climate_region in all_regions:
        states_in_region = climate_region_mapping[climate_region]
        region_period_data = period_data[period_data['StateName'].isin(states_in_region)].copy()
        if region_period_data.empty:
            continue
        
        for method in metrics:
            if method not in region_period_data.columns:
                continue
            
            # 处理harvard_ml的年份限制
            if method == 'harvard_ml':
                method_data = region_period_data[region_period_data['Year'] <= 2016].copy()
            else:
                method_data = region_period_data.copy()
            
            # 计算每个气候区-方法的人口加权数据
            yearly_data = pd.DataFrame()
            for year in years:
                year_data = method_data[method_data['Year'] == year].copy()
                if year_data.empty:
                    continue
                
                county_avg = year_data.groupby(['StateName', 'CountyName']).agg(
                    {method: 'mean', 'Population': 'first'}
                ).reset_index()
                
                if not county_avg.empty and 'Population' in county_avg.columns and not county_avg['Population'].isna().all():
                    pop_weighted = (county_avg[method] * county_avg['Population']).sum() / county_avg['Population'].sum()
                    yearly_data = pd.concat([yearly_data, pd.DataFrame({'Year': [year], 'Value': [pop_weighted]})])
            
            all_weighted_data.extend(yearly_data['Value'].tolist())  # 收集所有数据点
    
    # 计算全局范围
    if all_weighted_data:
        global_min_val = min(all_weighted_data)
        global_max_val = max(all_weighted_data)
        padding = (global_max_val - global_min_val) * 0.1  # 10%边距
        global_min_val -= padding
        global_max_val += padding
    else:
        print(f"警告: 时间段 {period} 无有效数据，跳过绘图")
        continue
    
    # ----------------------
    # 绘制每个气候区的图表
    # ----------------------
    for climate_region in tqdm(all_regions, desc="处理气候区", leave=False):
        states_in_region = climate_region_mapping[climate_region]
        region_period_data = period_data[period_data['StateName'].isin(states_in_region)].copy()
        
        if region_period_data.empty:
            print(f"警告: 气候区 {climate_region} 在 {period} 无数据，跳过")
            continue
        
        plt.figure(figsize=(12, 8))
        colors = ['#8c564b', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, method in enumerate(metrics):
            if method not in region_period_data.columns:
                continue
            
            display_name = method_display_names[method]
            marker = method_markers[method]
            
            # 处理harvard_ml的年份限制
            if method == 'harvard_ml':
                method_data = region_period_data[region_period_data['Year'] <= 2016].copy()
            else:
                method_data = region_period_data.copy()
            
            # 计算人口加权浓度
            yearly_data = pd.DataFrame()
            for year in years:
                year_data = method_data[method_data['Year'] == year].copy()
                if year_data.empty:
                    continue
                
                county_avg = year_data.groupby(['StateName', 'CountyName']).agg(
                    {method: 'mean', 'Population': 'first'}
                ).reset_index()
                
                if not county_avg.empty and 'Population' in county_avg.columns and not county_avg['Population'].isna().all():
                    pop_weighted = (county_avg[method] * county_avg['Population']).sum() / county_avg['Population'].sum()
                    yearly_data = pd.concat([yearly_data, pd.DataFrame({'Year': [year], method: [pop_weighted]})])
            
            if not yearly_data.empty:
                plt.plot(yearly_data['Year'], yearly_data[method], 
                         marker=marker, markersize=8, linestyle='-', linewidth=2, 
                         color=colors[i], label=display_name)
        
        # 设置统一的Y轴范围
        if y_limits[0] is not None and y_limits[1] is not None:
            plt.ylim(y_limits)
        else:
            plt.ylim(global_min_val, global_max_val)
        
        plt.title(f'{period}: {climate_region} O₃ Population-Weighted Time Series (2002-2019)', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Population-Weighted O₃ (ppm-hrs)', fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(years, rotation=45, fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        
        filename = f"{output_dir}/{period}_{region_mapping[climate_region]}_Ozone_PopWeighted_Timeseries.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

print("所有图表生成完成！")