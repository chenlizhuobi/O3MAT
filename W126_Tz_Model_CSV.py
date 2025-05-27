import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import numba


# 定义需要读取的列
usecols = ['ROW', 'COL', 'Timestamp', 'model']

# 定义每列的数据类型，减少内存使用
dtype = {
    'ROW': 'int32',
    'COL': 'int32',
    'Timestamp': 'object',
    'model': 'float32'
}


@numba.jit(nopython=True)
def calculate_weighted_values(values):
    return values / (1 + 4403 * np.exp(-126 * values))


def calculate_single_w126(group, col_name):
    values = group[col_name].values
    weighted_values = calculate_weighted_values(values)
    months = group['Month'].values
    years = group['Year'].values

    # 按年和月分组并计算每月的加权和
    unique_years_months, idx = np.unique(list(zip(years, months)), axis=0, return_index=True)
    monthly_weighted_sum = np.zeros(len(unique_years_months))
    for i, (year_month, start_idx) in enumerate(zip(unique_years_months, idx)):
        year, month = year_month
        mask = (years == year) & (months == month)
        monthly_weighted_sum[i] = np.sum(weighted_values[mask])

    # 定义特定的 3 个月组合
    specific_month_combinations = [(3, 4, 5), (4, 5, 6), (5, 6, 7), (6, 7, 8), (7, 8, 9), (8, 9, 10)]
    three_month_sums = []
    for start_month, middle_month, end_month in specific_month_combinations:
        start_mask = (unique_years_months[:, 1] == start_month)
        middle_mask = (unique_years_months[:, 1] == middle_month)
        end_mask = (unique_years_months[:, 1] == end_month)
        if np.sum(start_mask) > 0 and np.sum(middle_mask) > 0 and np.sum(end_mask) > 0:
            start_idx = np.where(start_mask)[0][0]
            middle_idx = np.where(middle_mask)[0][0]
            end_idx = np.where(end_mask)[0][0]
            three_month_sum = monthly_weighted_sum[start_idx] + monthly_weighted_sum[middle_idx] + monthly_weighted_sum[end_idx]
            three_month_sums.append(three_month_sum)

    if len(three_month_sums) > 0:
        return np.max(three_month_sums)
    return np.nan


def convert_all_to_local_time(df_data, timezone_df):
    """
    将所有数据的时间转换为本地时间，并添加年、月、小时列
    """
    # 合并数据
    merged_df = pd.merge(df_data, timezone_df, on=['ROW', 'COL'], how='left')
    # 转换时间戳
    merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp'])
    # 根据各个网格所在的时区将UTC转为相应的localtime, like ST = UTC + -5
    merged_df['local_time'] = merged_df['Timestamp'] + pd.to_timedelta(merged_df['gmt_offset'], unit='h')
    # 添加年、月、小时列
    merged_df['Year'] = merged_df['local_time'].dt.year
    merged_df['Month'] = merged_df['local_time'].dt.month
    merged_df['hour'] = merged_df['local_time'].dt.hour
    return merged_df


def calculate_w126_for_grid(grid_group, ozone_columns):
    row_num, col_num = grid_group[0]
    group = grid_group[1]
    w126_metrics = {'ROW': row_num, 'COL': col_num, 'Period': 'W126'}
    for col_name in ozone_columns:
        result = calculate_single_w126(group, col_name)
        w126_metrics[col_name] = result
    return w126_metrics


def calculate_w126_metric(df_data, save_path, year):
    ozone_columns = ['model']
    # 转换单位，ppbv to ppm
    df_data[ozone_columns] = df_data[ozone_columns] / 1000

    # 根据localtime筛选，注意修改年份时间
    df_year = df_data[(df_data['Year'] == year)]
    df_daytime = df_year[(df_year['hour'] >= 8) & (df_year['hour'] < 20)]

    grouped = df_daytime.groupby(['ROW', 'COL'])
    num_grids = len(grouped)
    print(f"总共需要处理 {num_grids} 个网格。")

    all_w126_metrics = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(calculate_w126_for_grid, grid_group, ozone_columns) for grid_group in grouped]
        for future in futures:
            all_w126_metrics.append(future.result())

    print(f"已处理完 {num_grids} 个网格。")

    w126_df = pd.DataFrame(all_w126_metrics)

    w126_output_file = f'{save_path}/{year}_W126_EQUATES.csv'
    w126_df[['ROW', 'COL', 'model', 'Period']].to_csv(w126_output_file, index=False)
    print(f"W126 指标数据已保存到: {w126_output_file}")
    return w126_output_file


def save_w126_metrics(save_path, file_path_base, timezone_file, years):
    print("开始保存 W126 指标...")
    timezone_df = pd.read_csv(timezone_file)

    print("文件读取完毕。")

    all_files = []
    for year in years:
        file_path = f'{file_path_base}/{year}_HourlyData_Limit.csv'
        print(f"开始读取 {file_path} ...")
        # 使用 dask 直接读取 CSV 文件
        dask_df = pd.read_csv(file_path, usecols=usecols, dtype=dtype)
        print(f"文件 {file_path} 读取完毕。")

        # 将所有数据转换为本地时间
        local_df = convert_all_to_local_time(dask_df, timezone_df)
        print(f"已完成 {year} 年的时间转换为本地时间。")

        # 计算 W126
        w126_df_file = calculate_w126_metric(local_df, save_path, year)
        print(f"{year} 年的 W126 指标保存完成.")
        all_files.append(w126_df_file)

    return all_files


if __name__ == "__main__":
    print("开始读取输入文件...")
    # 定义文件路径的基本部分
    file_path_base = "/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV"
    
    # 读取时区偏移表
    timezone_file = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/ROWCOLRegion_Tz_(CONUS+Ocean)_ST.csv'
    
    # 定义保存路径
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF"
    
    # 定义需要处理的年份列表
    years_to_process = [2019]

    # 调用函数计算指标并保存结果
    output_files = save_w126_metrics(save_path, file_path_base, timezone_file, years_to_process)
    print(f'W126 指标文件已保存到: {output_files}')