import pandas as pd

# 定义年份和文件路径
year = 2013
file_path = f"/backupdata/data_EPA/aq_obs/routine/{year}/AQS_hourly_data_{year}_LatLon.csv"

try:
    # 读取 CSV 文件
    df = pd.read_csv(file_path)
    
    # 检查 DataFrame 是否包含'site_id'列
    if 'site_id' in df.columns:
        # 剔除站点 400892001 的数据
        filtered_df = df[df['site_id'] != 400892001]
        
        # 保存处理后的数据到新文件
        output_file = f"/backupdata/data_EPA/aq_obs/routine/{year}/AQS_hourly_data_{year}_LatLon_filtered.csv"
        filtered_df.to_csv(output_file, index=False)
        
        print(f"已成功剔除站点 400892001 的数据，并保存到 {output_file}")
        print(f"原数据共有 {len(df)} 行，处理后剩余 {len(filtered_df)} 行")
    else:
        print(f"错误：数据中不包含'site_id'列。实际列名：{list(df.columns)}")
        
except FileNotFoundError:
    print(f"错误：文件 {file_path} 不存在")
except Exception as e:
    print(f"错误：处理文件时发生异常：{e}")