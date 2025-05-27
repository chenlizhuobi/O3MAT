import pandas as pd

# 读取输入文件
input_file = '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_O3MDA8_HourlyIntoDailyIntoMetrics.csv'
df = pd.read_csv(input_file)

# 定义 ROW 和 COL 的范围
rows = range(1, 300)
cols = range(1, 460)

# 定义需要处理的列
columns = ['vna_ozone', 'evna_ozone', 'avna_ozone', 'model']

# 创建所有可能的 ROW 和 COL 组合的 DataFrame
all_combinations = pd.DataFrame([(row, col) for row in rows for col in cols], columns=['ROW', 'COL'])

all_results = []

# 按 Period 分组处理
for period, group in df.groupby('Period'):
    # 将所有组合与当前分组合并
    merged = pd.merge(all_combinations, group, on=['ROW', 'COL'], how='left')

    # 处理合并后可能出现的重复列
    duplicate_cols = [col for col in merged.columns if col.endswith('_x') or col.endswith('_y')]
    for col in set([col[:-2] for col in duplicate_cols]):
        col_x = col + '_x'
        col_y = col + '_y'
        if col_x in merged.columns and col_y in merged.columns:
            merged[col] = merged[col_x].fillna(merged[col_y])
            merged.drop(columns=[col_x, col_y], inplace=True)

    # 确保最终的列名为 ROW 和 COL
    if 'ROW_x' in merged.columns:
        merged.rename(columns={'ROW_x': 'ROW'}, inplace=True)
        merged.drop(columns=['ROW_y'], errors='ignore', inplace=True)
    if 'COL_x' in merged.columns:
        merged.rename(columns={'COL_x': 'COL'}, inplace=True)
        merged.drop(columns=['COL_y'], errors='ignore', inplace=True)

    # 添加 Period 列
    merged['Period'] = period

    all_results.append(merged)

# 合并所有结果
final_result = pd.concat(all_results, ignore_index=True)

# 保存最终结果
output_file = f'{input_file}'
final_result.to_csv(output_file, index=False)
print(f'所有 Period 的结果已保存到 {output_file}')