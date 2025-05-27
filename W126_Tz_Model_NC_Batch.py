import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import numba
from pathlib import Path
import xarray as xr
from datetime import datetime

# 定义NC文件变量映射（确保与NC文件中的变量名一致）
NC_VARS = {
    'ROW': 'ROW',       # NC文件中的行变量名
    'COL': 'COL',       # NC文件中的列变量名
    'TIME': 'TSTEP',    # NC文件中的时间变量名
    'MODEL': 'O3'       # NC文件中的臭氧浓度变量名（需根据实际文件调整）
}

# 定义数据类型（确保与实际数据一致）
dtype = {
    'ROW': 'int32',     # 行号数据类型
    'COL': 'int32',     # 列号数据类型
    'Timestamp': 'object',  # 时间戳字符串类型
    'model': 'float32'  # 臭氧浓度数据类型
}

# 固定参数配置
START_MONTH = 3       # 处理起始月份
END_MONTH = 10        # 处理结束月份
HOUR_RANGE = (8, 20)  # 白天小时范围（8:00-20:00）

@numba.jit(nopython=True)
def calculate_weighted_values(values):
    """计算加权值（W126指数核心公式）"""
    return values / (1 + 4403 * np.exp(-126 * values))

@numba.jit(nopython=True)
def sliding_window_sum(arr, window_size):
    """滑动窗口求和（用于计算连续三月累计值）"""
    n = len(arr)
    result = np.empty(n - window_size + 1)
    for i in range(n - window_size + 1):
        result[i] = np.sum(arr[i:i + window_size])
    return result

def calculate_single_w126(group, col_name):
    """计算单个网格点的W126指数"""
    values = group[col_name].values
    weighted_values = calculate_weighted_values(values)
    months = group['Month'].values
    years = group['Year'].values

    # 提取唯一的年-月组合
    unique_years_months = np.unique(list(zip(years, months)), axis=0)
    monthly_weighted_sum = np.zeros(len(unique_years_months))
    
    # 计算每月加权和
    for i, (year, month) in enumerate(unique_years_months):
        mask = (years == year) & (months == month)
        monthly_weighted_sum[i] = np.sum(weighted_values[mask])

    # 定义需要计算的连续三月组合（如MAM, AMJ等）
    specific_month_combinations = [
        (3, 4, 5), (4, 5, 6), (5, 6, 7),
        (6, 7, 8), (7, 8, 9), (8, 9, 10)
    ]
    
    three_month_sums = []
    for start, mid, end in specific_month_combinations:
        # 查找各月份在唯一列表中的索引
        start_idx = np.where((unique_years_months[:, 1] == start))[0]
        mid_idx = np.where((unique_years_months[:, 1] == mid))[0]
        end_idx = np.where((unique_years_months[:, 1] == end))[0]
        
        # 检查是否存在所有三个月的数据
        if start_idx.size > 0 and mid_idx.size > 0 and end_idx.size > 0:
            total = monthly_weighted_sum[start_idx[0]] + \
                    monthly_weighted_sum[mid_idx[0]] + \
                    monthly_weighted_sum[end_idx[0]]
            three_month_sums.append(total)
    
    return np.max(three_month_sums) if three_month_sums else np.nan

def convert_all_to_local_time(df_data, timezone_df):
    """将UTC时间转换为本地时间"""
    merged_df = pd.merge(df_data, timezone_df, on=['ROW', 'COL'], how='left')
    merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp'])
    merged_df['local_time'] = merged_df['Timestamp'] + pd.to_timedelta(merged_df['gmt_offset'], unit='h')
    merged_df['Year'] = merged_df['local_time'].dt.year
    merged_df['Month'] = merged_df['local_time'].dt.month
    merged_df['hour'] = merged_df['local_time'].dt.hour
    return merged_df

def calculate_w126_for_grid(grid_group, ozone_columns):
    """计算单个网格的W126指标"""
    (row_num, col_num), group = grid_group  # 解包分组数据
    w126_metrics = {
        'ROW': row_num,
        'COL': col_num,
        'Period': 'W126'  # 固定周期标识
    }
    
    for col_name in ozone_columns:
        result = calculate_single_w126(group, col_name)
        w126_metrics[col_name] = result  # 填充臭氧浓度结果
    
    return w126_metrics

def calculate_w126_metric(df_data, ozone_columns, target_year):
    """计算W126指数主函数"""
    # 单位转换：ppbv转ppm（1 ppm = 1000 ppbv）
    df_data[ozone_columns] = df_data[ozone_columns] / 1000
    
    # 筛选目标年份数据
    df_year = df_data[df_data['Year'] == target_year].copy()
    
    # 筛选白天时间段（HOUR_RANGE定义的小时范围）
    df_daytime = df_year[
        (df_year['hour'] >= HOUR_RANGE[0]) &
        (df_year['hour'] < HOUR_RANGE[1])
    ]
    
    # 按网格分组（ROW, COL）
    grouped = df_daytime.groupby(['ROW', 'COL'])
    
    # 并行计算每个网格的W126指标
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(calculate_w126_for_grid, group, ozone_columns) for _, group in grouped]
        results = [future.result() for future in futures]
    
    return pd.DataFrame(results)

def generate_nc_filepaths(base_path, year, month_range):
    """生成NC文件路径列表（自动过滤不存在的文件）"""
    filepaths = []
    start_month, end_month = month_range
    for month in range(start_month, end_month + 1):
        filename = f"EQUATES_COMBINE_ACONC_O3_{year}{month:02d}.nc"
        filepath = Path(base_path) / filename
        if filepath.exists():
            filepaths.append(str(filepath))
        else:
            print(f"警告：文件 {filepath} 不存在，跳过该月份")
    return filepaths

def process_year(year, base_nc_path, timezone_file, output_dir, month_range=(START_MONTH, END_MONTH)):
    """处理单个年份的完整流程"""
    print(f"\n开始处理年份：{year}")
    
    # 生成NC文件路径列表
    nc_files = generate_nc_filepaths(base_nc_path, year, month_range)
    if not nc_files:
        print(f"警告：年份 {year} 无有效NC文件，跳过处理")
        return
    
    all_dfs = []
    for nc_path in nc_files:
        try:
            print(f"处理文件：{nc_path}")
            with xr.open_dataset(nc_path) as ds:
                # 提取时间数据（转换为datetime对象）
                time_values = ds[NC_VARS['TIME']].values
                timestamps = pd.DatetimeIndex(time_values).strftime('%Y-%m-%d %H:%M:%S')
                
                # 提取模型数据（维度：TIME, VAR, ROW, COL）
                model_data = ds[NC_VARS['MODEL']].values
                n_times, n_vars, n_rows, n_cols = model_data.shape
                
                # 提取行/列坐标（假设为一维数组，如0-based索引）
                rows_1d = ds[NC_VARS['ROW']].values
                cols_1d = ds[NC_VARS['COL']].values
                
                # 生成二维网格坐标并展平
                rows_2d, cols_2d = np.meshgrid(rows_1d, cols_1d, indexing='ij')  # 注意indexing参数根据实际情况调整
                row_flat = rows_2d.flatten()
                col_flat = cols_2d.flatten()
                
                # 验证网格大小
                grid_size = n_rows * n_cols
                assert len(row_flat) == grid_size and len(col_flat) == grid_size, "网格坐标展平错误"
                
                # 展平模型数据（按时间-变量-行-列顺序）
                model_flat = model_data.reshape(n_times * n_vars, grid_size).flatten()
                
                # 生成重复的时间戳和坐标（每个时间-变量对应所有网格点）
                timestamps_repeated = np.tile(timestamps, grid_size)
                row_repeated = np.repeat(row_flat, n_times * n_vars)
                col_repeated = np.repeat(col_flat, n_times * n_vars)
                
                # 验证数组长度一致性
                assert len(timestamps_repeated) == len(model_flat) == len(row_repeated) == len(col_repeated), "数组长度不一致"
                
                # 创建DataFrame并添加到列表
                df = pd.DataFrame({
                    'ROW': row_repeated.astype(dtype['ROW']),
                    'COL': col_repeated.astype(dtype['COL']),
                    'Timestamp': timestamps_repeated,
                    'model': model_flat.astype(dtype['model'])
                })
                all_dfs.append(df)
                print(f"  数据加载完成：时间步={n_times}, 变量={n_vars}, 网格点={grid_size}")
        
        except Exception as e:
            print(f"文件 {nc_path} 处理失败：{str(e)}，跳过")
            continue
    
    if not all_dfs:
        print(f"年份 {year} 无有效数据")
        return
    
    # 合并所有月份数据
    df_data = pd.concat(all_dfs, ignore_index=True)
    print(f"合并后数据量：{len(df_data)} 条记录")
    
    # 加载时区数据并转换为本地时间
    timezone_df = pd.read_csv(timezone_file, dtype={'ROW': 'int32', 'COL': 'int32'})
    local_df = convert_all_to_local_time(df_data, timezone_df)
    
    # 计算W126指标（确保传递正确的列名）
    w126_df = calculate_w126_metric(local_df, ['model'], year)
    
    # 验证结果列是否存在
    required_columns = ['ROW', 'COL', 'model', 'Period']
    if not set(required_columns).issubset(w126_df.columns):
        missing = set(required_columns) - set(w126_df.columns)
        raise ValueError(f"结果缺少关键列：{missing}")
    
    # 保存结果到CSV
    output_file = Path(output_dir) / f"{year}_W126_EQUATES_{month_range[0]}-{month_range[1]}.csv"
    w126_df[required_columns].to_csv(output_file, index=False)
    print(f"结果已保存：{output_file}，包含 {len(w126_df)} 个网格点")
    return output_file

if __name__ == "__main__":
    # 配置参数（需根据实际路径修改）
    BASE_NC_PATH = Path("/backupdata/data_EPA/EQUATES/o3_hourly_files")  # NC文件根目录
    TIMEZONE_FILE = "/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/ROWCOLRegion_Tz_(CONUS+Ocean)_ST.csv"  # 时区文件路径
    OUTPUT_DIR = "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF"  # 输出目录
    
    # 创建输出目录（若不存在）
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 处理年份列表（可扩展为多个年份）
    YEARS = [2011]
    
    # 批量处理
    for year in YEARS:
        try:
            process_year(year, BASE_NC_PATH, TIMEZONE_FILE, OUTPUT_DIR)
        except Exception as e:
            print(f"年份 {year} 处理失败：{str(e)}")
    
    print("\n所有年份处理完成！")