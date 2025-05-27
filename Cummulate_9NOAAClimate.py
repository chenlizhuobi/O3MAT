import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 设置字体为新罗马
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取气候区域映射文件 - 仅包含网格基本信息，没有Period
region_map = pd.read_csv('/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/2011_299*459_9ClimateRegions.csv')
region_map = region_map[['ROW', 'COL', 'ClimateRegion']].dropna()
region_map['ClimateRegion'] = region_map['ClimateRegion'].astype(int)

# 定义需要绘图的指标（方法）
metrics = ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone', 'ds_ozone', 'harvard_ml']

# 变量名映射（替换为更友好的显示名称）
method_display_names = {
   'model': 'EQUATES',  # 修改为EQUATES
    'vna_ozone': 'VNA',
    'evna_ozone': 'eVNA',
    'avna_ozone': 'aVNA',
    'ds_ozone': 'Downscaler',
    'harvard_ml': 'Harvard ML',
}

# 定义不同方法的线条样式
method_linestyles = {
   'model': '--',      # EQUATES为虚线
    'vna_ozone': '-',  # VNA为实线
    'evna_ozone': '--', # eVNA为虚线
    'avna_ozone': '-', # aVNA为实线
    'ds_ozone': '--',   # Downscaler为虚线
    'harvard_ml': '-', # Harvard ML为实线
}

# 定义不同方法的颜色
method_colors = {
   'model': '#8c564b',
    'vna_ozone': '#1f77b4',
    'evna_ozone': '#ff7f0e',
    'avna_ozone': '#2ca02c',
    'ds_ozone': '#d62728',
    'harvard_ml': '#9467bd',
}

# 自定义Y轴范围（累积分布函数的Y轴范围是0到100）
y_limits = (0, 100)

# 为每个Period指定X轴范围，如果为None则自动计算
period_x_limits = {
    'W126': (0, 52),  # 特殊Period W126的X轴范围
    'top-10': (0, 105),  # 特殊Period top-10的X轴范围
    'DJF': None,
    'MAM': None,
    'JJA': None,
    'SON': None,
    'Annual': None,
    'Apr-Sep': None
}

# 定义Periods
periods = ['DJF']
periods = ['W126'] 
# 创建输出目录
output_dir = '/DeepLearning/mnt/shixiansheng/data_fusion/output/CumulativeDistributionPlots'
os.makedirs(output_dir, exist_ok=True)

# 为每个Period生成累积分布函数图
print("正在生成累积分布函数图...")
for period in tqdm(periods, desc="处理时间段"):
    # 读取原始数据
    years = [2002, 2003]  # 可以指定任意年份列表
    years = [2002, 2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]  # 可以指定任意年份列表
    
    # 为所有年份的数据创建一个DataFrame，仅用于计算X轴范围
    print(f"正在为 {period} 时间段计算X轴范围...")
    all_years_data = pd.DataFrame()
    
    for year in tqdm(years, desc="读取数据计算X轴范围", leave=False):
        # 根据Period决定使用哪个数据源
        if period == 'W126':
            file_path = f'/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/{year}_W126_AtF.csv'
            # file_path = f'/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/{year}_W126_ST_Limit.csv'
        else:
            file_path = f'/DeepLearning/mnt/shixiansheng/data_fusion/output/DailyData_WithoutCV/{year}_Data_WithoutCV_Metrics.csv'
        
        # 检查文件名是否包含AtF
        use_atf_data = 'AtF' in file_path
        
        try:
            data = pd.read_csv(file_path)
            data['Year'] = year
            # 合并气候区域信息
            data = data.merge(region_map, on=['ROW', 'COL'], how='left')
            all_years_data = pd.concat([all_years_data, data], ignore_index=True)
        except FileNotFoundError:
            print(f"  - 未找到 {file_path} 文件，跳过 {year} 年数据")
    
    # 筛选出特定Period的数据
    period_all_years_data = all_years_data[all_years_data['Period'] == period].copy()
    
    if period_all_years_data.empty:
        print(f"警告: 时间段 {period} 没有数据，跳过绘图")
        continue
    
    # 只取ClimateRegion不为-999的所有网格作为USA的数据
    usa_all_years_data = period_all_years_data[period_all_years_data['ClimateRegion'] != -999]
    
    # 计算所有方法的最小值和最大值，用于统一X轴范围
    min_value = float('inf')
    max_value = float('-inf')
    
    methods = ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone', 'ds_ozone', 'harvard_ml']
    
    for method in methods:
        if method in usa_all_years_data.columns:
            # 特殊处理 harvard_ml：仅使用2002-2016年的数据
            if method == 'harvard_ml':
                method_data = usa_all_years_data[usa_all_years_data['Year'] <= 2016].copy()
            else:
                method_data = usa_all_years_data.copy()
            
            if not method_data.empty:
                method_min = method_data[method].min()
                method_max = method_data[method].max()
                min_value = min(min_value, method_min)
                max_value = max(max_value, method_max)
    
    # 设置X轴范围（如果没有指定）
    if period in period_x_limits and period_x_limits[period] is not None:
        x_limits = period_x_limits[period]
    else:
        # 预留一定空间
        if min_value != float('inf') and max_value != float('-inf'):
            padding = (max_value - min_value) * 0.1
            x_limits = (min_value - padding, max_value + padding)
        else:
            x_limits = None
    
    print(f"X轴范围已计算完成: {x_limits}")
    
    # 对每个年份单独处理
    for year in tqdm(years, desc=f"处理{period}各年份", leave=False):
        print(f"\n开始处理 {year} 年数据...")
        
        # 读取当前年份的数据
        if period == 'W126':
            file_path = f'/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/{year}_W126_AtF.csv'
            # file_path = f'/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/{year}_W126_ST_Limit.csv'
        else:
            file_path = f'/DeepLearning/mnt/shixiansheng/data_fusion/output/DailyData_WithoutCV/{year}_Data_WithoutCV_Metrics.csv'
        
        # 检查文件名是否包含AtF
        use_atf_data = 'AtF' in file_path
        
        try:
            data = pd.read_csv(file_path)
            data['Year'] = year
            # 合并气候区域信息
            data = data.merge(region_map, on=['ROW', 'COL'], how='left')
            
            # 筛选出特定Period的数据
            period_data = data[data['Period'] == period].copy()
            
            if period_data.empty:
                print(f"警告: {year}年 {period} 时间段没有数据，跳过绘图")
                continue
            
            # 只取ClimateRegion不为-999的所有网格作为USA的数据
            usa_data = period_data[period_data['ClimateRegion'] != -999]
            
            if usa_data.empty:
                print(f"警告: {year}年 {period} 时间段没有有效数据，跳过绘图")
                continue
            
            print(f"开始绘制 {year} 年 {period} 时间段的累积分布函数图...")
            
            plt.figure(figsize=(12, 8))
            
            # 绘制各方法的累积分布函数
            for method in methods:
                if method in usa_data.columns:
                    # 特殊处理 harvard_ml：仅使用2002-2016年的数据
                    if method == 'harvard_ml' and year > 2016:
                        print(f"  - 跳过 {method_display_names[method]} 指标 (2016年后无数据)")
                        continue
                    
                    print(f"  - 正在绘制 {method_display_names[method]} 指标...")
                    
                    # 获取显示名称、线条样式和颜色
                    display_name = method_display_names.get(method, method)
                    linestyle = method_linestyles.get(method, '-')
                    color = method_colors.get(method, '#000000')
                    
                    # 对数据进行排序
                    sorted_data = usa_data.sort_values(by=method)[method]
                    # 计算累积分布
                    cdf = (sorted_data.reset_index(drop=True).index + 1) / len(sorted_data)
                    
                    # 绘制累积分布函数
                    plt.plot(sorted_data, cdf * 100, 
                             linestyle=linestyle, 
                             linewidth=2, 
                             color=color, 
                             label=display_name)
            
            # 设置Y轴范围
            plt.ylim(y_limits)
            
            # 设置X轴范围
            if x_limits:
                plt.xlim(x_limits)
            
            # 设置图表标题和轴标签，添加年份信息
            plt.title(f'Cumulative Distribution of O₃ in USA ({period}, {year})', fontsize=16)
            plt.xlabel('O₃ (ppbv)', fontsize=14)
            plt.ylabel('Cumulative Probability (%)', fontsize=14)
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='best', fontsize=12)
            plt.tight_layout()
            
            # 保存图表，文件名添加年份信息和AtF后缀（仅当文件名包含AtF时）
            atf_suffix = "_AtF" if use_atf_data else ""
            filename = f"{output_dir}/{year}_{period}{atf_suffix}_CumulativeDistributionPlot.png"
            print(f"正在保存 {year} 年图表: {filename}")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"{year} 年图表保存完成!")
            
        except FileNotFoundError:
            print(f"未找到 {file_path} 文件，跳过 {year} 年数据")
    
print("\n所有累积分布函数图生成完成！")
