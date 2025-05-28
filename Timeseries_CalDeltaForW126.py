import pandas as pd

# 定义可输入的多个基准年份
base_years = [2002]  # 你可以在这里添加更多基准年份，例如 [2002, 2005]

# 定义要处理的年份范围，从基准年份后的第一年开始到 2019 年
all_years = [2016]

# 定义要计算变化量的指标
common_metrics = ['model','vna_ozone', 'evna_ozone', 'avna_ozone']
metrics_2016_and_before = common_metrics + ['harvard_ml']

for base_year in base_years:
    try:
        base_data = pd.read_csv(f'/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/{base_year}_W126_AtF.csv')
    except FileNotFoundError:
        print(f"未找到 {base_year}_W126_AtF.csv 文件。")
        continue

    for year in all_years:
        if year <= base_year:
            continue
        try:
            # 修正：使用当前年份的文件路径
            current_data = pd.read_csv(f'/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/{year}_W126_AtF.csv')

            # 检查 ROW、COL 和 Period 是否一致
            key_columns = ['ROW', 'COL', 'Period']
            if not (base_data[key_columns].equals(current_data[key_columns])):
                raise ValueError(f"年份 {year} 与基准年份 {base_year} 的 ROW、COL 或 Period 列数据不一致。")
            metrics = common_metrics

            change_data = current_data.copy()
            for metric in metrics:
                change_data[metric] = (current_data[metric] - base_data[metric])

            # 确保保留 Period 列并按原顺序
            columns_to_keep = ['Period'] + [col for col in current_data.columns if col in ['ROW', 'COL'] + metrics]
            change_data = change_data[columns_to_keep]

            # 保存结果到单独的 CSV 文件
            output_file = f'/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/{year}-{base_year}_W126_AtF.csv'
            change_data.to_csv(output_file, index=False)
            print(f"{year} 年相对于 {base_year} 年的臭氧指标绝对变化量已保存到 {output_file} 文件中。")
        except FileNotFoundError:
            print(f"未找到 {year}_W126_AtF.csv 文件。")
        except ValueError as ve:
            print(ve)