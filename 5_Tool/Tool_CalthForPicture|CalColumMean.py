# import pandas as pd
# import numpy as np
# import itertools

# # 数据文件路径
# data_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Timeseries/thiel_sen_slope_results.csv'

# # 读取数据文件
# df = pd.read_csv(data_path)

# # # 指定要提取数据的 Period 值
# target_periods = ['DJF', 'MAM', 'JJA', 'SON', 'Annual', 'Apr-Sep','top-10']

# # 指定要提取的列
# target_columns = ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone','ds_ozone','harvard_ml']
# # target_columns = ['Missing']

# # 提取对应 Period 和列的数据
# extracted_data = df[df['Period'].isin(target_periods)][target_columns]

# # 将指定列的数据合并为一个一维数组
# combined_data = extracted_data.values.flatten()

# # 定义默认的最大值和最小值（可根据需要修改）
# default_max_value = None
# default_min_value = None

# # 计算 99.5 分位值
# vmax_conc = (
#     np.nanpercentile(combined_data, 99.5)
#     if default_max_value is None
#     else default_max_value
# )

# # 计算 0.5 分位值
# vmin_conc = (
#     np.nanpercentile(combined_data, 0.5)
#     if default_min_value is None
#     else default_min_value
# )

# # 计算整体数据的最大值和最小值
# overall_max = np.nanmax(combined_data)
# overall_min = np.nanmin(combined_data)

# print("整体数据的 99.5 分位值：", vmax_conc)
# print("整体数据的 0.5 分位值：", vmin_conc)
# print("整体数据的最大值：", overall_max)
# print("整体数据的最小值：", overall_min)

# # 生成所有可能的列对
# variables = ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone', 'ds_ozone', 'harvard_ml']
# comparisons = list(itertools.combinations(variables, 2))

# # 存储所有列对的差值
# all_differences = []

# # 计算每对列的差值并存储
# for col1, col2 in comparisons:
#     diff = extracted_data[col1] - extracted_data[col2]
#     all_differences.extend(diff)

# # 将差值列表转换为 NumPy 数组
# all_differences = np.array(all_differences)

# # 计算所有差值的 99.5 分位值
# diff_vmax_conc = (
#     np.nanpercentile(all_differences, 99.5)
#     if default_max_value is None
#     else default_max_value
# )

# # 计算所有差值的 0.5 分位值
# diff_vmin_conc = (
#     np.nanpercentile(all_differences, 0.5)
#     if default_min_value is None
#     else default_min_value
# )

# # 计算所有差值数据的最大值和最小值
# diff_max = np.nanmax(all_differences)
# diff_min = np.nanmin(all_differences)

# print("所有列对差值的 99.5 分位值：", diff_vmax_conc)
# print("所有列对差值的 0.5 分位值：", diff_vmin_conc)
# print("所有列对差值的最大值：", diff_max)
# print("所有列对差值的最小值：", diff_min)

# import pandas as pd


# def calculate_stats(df, column_name, periods=None, period_column='Period'):
#     """
#     此函数用于计算指定列的统计信息，可根据 Period 列进行筛选。

#     :param df: 输入的 DataFrame
#     :param column_name: 要计算统计信息的列名
#     :param periods: 可选，指定要筛选的 Period 值列表，默认为 None
#     :param period_column: 可选，Period 列的列名，默认为 'Period'
#     :return: 包含 Min、Max 和 Mean 的统计信息
#     """
#     if periods is not None and period_column in df.columns:
#         df = df[df[period_column].isin(periods)]

#     min_value = df[column_name].min()
#     max_value = df[column_name].max()
#     mean_value = df[column_name].mean()

#     return round(min_value,2), round(max_value,2), round(mean_value,2)


# # 读取 CSV 文件
# file_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/mean_data_by_site.csv'
# file_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/rn_filtered_data_W126.csv'
# # file_path = '/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011_MissingDays.csv'
# try:
#     df = pd.read_csv(file_path)
#     # 要计算统计信息的列名
#     column_name = 'r_n'

#     # 计算统计信息，不指定 Period
#     min_value, max_value, mean_value = calculate_stats(df, column_name)
#     print(f"All Rows - Min= {min_value}, Max= {max_value}, Mean= {mean_value}")

#     # # 定义要计算的多个 Period
#     # periods = ['DJF', 'MAM', 'JJA', 'SON', 'Annual', 'Apr - Sep']

#     # # 针对每个 Period 分别计算统计信息
#     # for period in periods:
#     #     min_value, max_value, mean_value = calculate_stats(df, column_name, periods=[period])
#     #     print(f"Period {period} - Min= {min_value}, Max= {max_value}, Mean= {mean_value}")

# except FileNotFoundError:
#     print(f"错误: 文件 {file_path} 未找到。")
# except Exception as e:
#     print(f"错误: 发生了未知错误: {e}")
    

import pandas as pd
import numpy as np

# 初始化一个空的 DataFrame 用于存储结果
result_df = pd.DataFrame(columns=['Year', 'Period', '0.5th', '99.5th', 'Min', 'Max'])

# 定义要处理的年份范围
years = [2002,2003,2004,2005,2006,2007,2008,2009,2010,
        2011,2012,2013,2014,2015,2016,2017,2018,2019]

# 定义要提取的列
target_columns = ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone']
target_columns = ['SD']

for year in years:
    # data_path = f'/DeepLearning/mnt/shixiansheng/data_fusion/output/DailyData_WithoutCV_Delta/{year}-2002_Data_WithoutCV_Metrics.csv'
    data_path = f'/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/{year}_CVSD_HourlyMetrics.csv'
    data_path = f'/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/{year}_CVSD_HourlyMetrics_AtF.csv'
    try:
        # 读取数据文件
        df = pd.read_csv(data_path)

        # 处理 not top-10 的 Period
        non_top_10_periods = [period for period in df['Period'].unique() if period != 'top-10']
        if non_top_10_periods:
            available_columns = [col for col in target_columns if col in df.columns]
            extracted_data = df[df['Period'].isin(non_top_10_periods)][available_columns]
            combined_data = extracted_data.values.flatten()
            vmin_conc = np.nanpercentile(combined_data, 0.5)
            vmax_conc = np.nanpercentile(combined_data, 99.5)
            overall_min = np.nanmin(combined_data)
            overall_max = np.nanmax(combined_data)
            result_df = pd.concat([result_df, pd.DataFrame({
                'Year': [year],
                'Period': ['not top-10'],
                '0.5th': [vmin_conc],
                '99.5th': [vmax_conc],
                'Min': [overall_min],
                'Max': [overall_max]
            })], ignore_index=True)

        # 处理 top-10 Period
        if 'top-10' in df['Period'].values:
            available_columns = [col for col in target_columns if col in df.columns]
            extracted_data = df[df['Period'] == 'top-10'][available_columns]
            combined_data = extracted_data.values.flatten()
            vmin_conc = np.nanpercentile(combined_data, 0.5)
            vmax_conc = np.nanpercentile(combined_data, 99.5)
            overall_min = np.nanmin(combined_data)
            overall_max = np.nanmax(combined_data)
            result_df = pd.concat([result_df, pd.DataFrame({
                'Year': [year],
                'Period': ['top-10'],
                '0.5th': [vmin_conc],
                '99.5th': [vmax_conc],
                'Min': [overall_min],
                'Max': [overall_max]
            })], ignore_index=True)

    except FileNotFoundError:
        print(f"未找到 {data_path} 文件，跳过该年份。")

# 按 Period 和 Year 排序
result_df = result_df.sort_values(by=['Period', 'Year'])

# 计算每个 Period 下各年份的均值
mean_df = result_df.groupby('Period')[['0.5th', '99.5th', 'Min', 'Max']].mean()
mean_df['Year'] = 'Mean'
mean_df.reset_index(inplace=True)

# 将均值行插入到每个 Period 的末尾
new_result_df = pd.DataFrame()
for period in result_df['Period'].unique():
    period_df = result_df[result_df['Period'] == period]
    mean_row = mean_df[mean_df['Period'] == period]
    new_result_df = pd.concat([new_result_df, period_df, mean_row], ignore_index=True)

# 输出结果表格
print(new_result_df)

# 保存结果到 CSV 文件（可选）
new_result_df.to_csv('/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/PictureRadius_CV_ForW126AtF.csv', index=False)
new_result_df.to_csv('/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/PictureRadius_SD_ForW126AtF.csv', index=False)
    
        