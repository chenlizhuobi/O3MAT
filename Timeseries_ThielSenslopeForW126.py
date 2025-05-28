import pandas as pd
from scipy.stats import theilslopes
from tqdm import tqdm

# 定义要处理的年份范围
years = range(2002, 2020)

# 初始化一个空的 DataFrame 用于存储所有数据
all_data = pd.DataFrame()

# 循环读取每年的数据，添加进度条
print("正在读取每年的数据...")
for year in tqdm(years):
    file_path = f'/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/{year}_W126_AtF.csv'
    try:
        data = pd.read_csv(file_path)
        data['Year'] = year
        all_data = pd.concat([all_data, data], ignore_index=True)
    except FileNotFoundError:
        print(f"未找到 {file_path} 文件。")

# 定义要计算泰尔 - 森斜率的变量（移除了 harvard_ml）
variables = ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone']

# 定义不同的 Period
periods = ['W126']

# 初始化一个空的 DataFrame 用于存储最终结果
final_result = pd.DataFrame()

# 循环遍历每个 Period，添加进度条
print("正在计算每个 Period 的泰尔 - 森斜率...")
for period in tqdm(periods):
    period_data = all_data[all_data['Period'] == period]
    
    # 计算所有变量的泰尔 - 森斜率
    result_df = period_data.groupby(['ROW', 'COL']).apply(
        lambda group: pd.Series({var: theilslopes(group[var], group['Year'])[0] for var in variables})
    ).reset_index()
    
    result_df['Period'] = period
    final_result = pd.concat([final_result, result_df], ignore_index=True)

# 保存结果到 CSV 文件
final_result.to_csv('/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_Timeseries/thiel_sen_slope_results.csv', index=False)

print("泰尔 - 森斜率计算完成，结果已保存到 thiel_sen_slope_results.csv 文件中。")