import pandas as pd
import os

# 指定年份列表
years = [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

# 假设参与计算的列，你需要根据实际情况修改
# colums_W126
columns_to_calculate = ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone']

# 输出文件夹路径
output_folder = '/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF'
# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 遍历每个年份
for year in years:
    # 数据文件路径
    data_path = f"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/{year}_W126_ST_Limit.csv"
    data_path = f"/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/{year}_W126_AtF.csv"

    try:
        # 读取数据
        df = pd.read_csv(data_path)

        # 用于存储当前年份结果的数据列表
        results = []

        # 按 Period 分组进行计算
        for period, group in df.groupby('Period'):
            # 按 ROW 和 COL 分组计算标准偏差和变异系数
            for (row, col), sub_group in group.groupby(['ROW', 'COL']):
                # 提取参与计算的列数据
                calc_data = sub_group[columns_to_calculate]
                # 检查是否全为 NaN
                if calc_data.isna().all().all():
                    std_dev = float('nan')
                    mean_value = float('nan')
                else:
                    # 计算标准偏差
                    std_dev = calc_data.std(axis=1).mean()
                    # 计算均值
                    mean_value = calc_data.mean(axis=1).mean()

                # 计算变异系数
                cv = (std_dev / mean_value) * 100 if pd.notna(mean_value) and mean_value != 0 else float('nan')

                results.append([row, col, std_dev, cv, period, year])

        # 创建当前年份的结果数据框
        result_df = pd.DataFrame(results, columns=['ROW', 'COL', 'SD', 'CV', 'Period', 'Year'])

        # 保存当前年份的结果到文件
        output_file = os.path.join(output_folder, f'{year}_CVSD_HourlyMetrics_AtF.csv')
        result_df.to_csv(output_file, index=False)
        print(f"年份 {year} 的结果已保存到 {output_file}")
    except FileNotFoundError:
        print(f"文件 {data_path} 未找到，跳过该年份。")
    except Exception as e:
        print(f"处理年份 {year} 的数据时出现错误: {e}")
    