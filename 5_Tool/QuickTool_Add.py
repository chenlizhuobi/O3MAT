import pandas as pd

# 读取CSV文件
file_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/2011_299*459_CountyRegions.csv'
df = pd.read_csv(file_path)

# 统计ClimateRegion列中不等于-999的行数
valid_rows = df[df['StateRegionId'] != -999]
count = len(valid_rows)

print(f"ClimateRegion不等于-999的行数: {count}")    