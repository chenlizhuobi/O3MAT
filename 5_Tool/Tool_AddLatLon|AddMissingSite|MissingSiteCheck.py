import pandas as pd


def merge_station_data(station_file, monitor_file, output_file, comparison_file):
    # 读取完整的监测数据，并过滤掉 O3 为 -999 的行
    df_monitor = pd.read_csv(monitor_file, dtype={'site_id': str})
    original_monitor_site_ids = df_monitor['site_id'].unique()  # 去除前的监测数据站点
    df_monitor = df_monitor[df_monitor['O3'] != -999]  # 去除 O3 为 -999 的行
    filtered_monitor_site_ids = df_monitor['site_id'].unique()  # 去除后的监测数据站点

    # 读取站点数据，获取唯一站点 ID
    df_station = pd.read_csv(station_file, header=None, usecols=[0], names=['Site'], dtype={'Site': str})
    original_station_site_ids = df_station['Site'].unique()  # 站点数据中的唯一站点
    filtered_station_site_ids = df_station['Site'].unique()  # 假设站点文件不含 -999，保持不变

    # 打印去除 O3 == -999 后，站点数量的变化
    print(f"去除 O3 == -999 后，监测文件中的唯一站点数量：{len(filtered_monitor_site_ids)}")
    print(f"站点文件中的唯一站点数量：{len(filtered_station_site_ids)}")

    # 比较原始和过滤后的唯一站点 ID
    print(f"原始监测文件中的站点数量：{len(original_monitor_site_ids)}")
    print(f"原始站点文件中的站点数量：{len(original_station_site_ids)}")

    # 继续读取监测数据（已经过滤了 O3 为 -999 的行）
    df_monitor = pd.read_csv(monitor_file, dtype={'site_id': str})
    df_monitor = df_monitor[df_monitor['site_id'].isin(filtered_monitor_site_ids)]

    # 读取站点数据，但只保留唯一站点对应的数据，指定数据类型为字符串
    df_station = pd.read_csv(station_file, header=None,
                             names=['Site', 'POC', 'Date', 'Lat', 'Lon', 'Conc'],
                             dtype={'Site': str})
    df_station = df_station[df_station['Site'].isin(filtered_station_site_ids)]

    # 对于每个站点，取第一个出现的经纬度信息
    df_station = df_station.groupby('Site').first().reset_index()

    # 合并数据，根据站点 ID
    merged_df = pd.merge(df_monitor, df_station, left_on='site_id',
                         right_on='Site', how='left')

    # 选择需要的列
    final_df = merged_df[['site_id', 'POCode', 'dateon', 'dateoff', 'O3', 'Lat', 'Lon']]

    # 去除没有经纬度的站点
    final_df = final_df.dropna(subset=['Lat', 'Lon'])

    # 保存最终结果
    final_df.to_csv(output_file, index=False)
    print("数据合并完成，结果已保存到", output_file)

    # 比较缺失和多出的站点
    missing_sites = set(filtered_station_site_ids) - set(filtered_monitor_site_ids)
    extra_sites = set(filtered_monitor_site_ids) - set(filtered_station_site_ids)

    # 将比较结果保存到 CSV
    comparison_df = pd.DataFrame({
        'Missing Site ID in merged data': list(missing_sites),
        'Extra Site ID in merged data': list(extra_sites)
    })
    comparison_df.to_csv(comparison_file, index=False)
    print(f"缺失和多余的站点信息已保存到 {comparison_file}")


if __name__ == "__main__":
    monitor_file = r"/backupdata/data_EPA/EQUATES/2011_Hour_Data/AQS_hourly_data_2011.csv"
    station_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv"
    output_file = r"/backupdata/data_EPA/EQUATES/2011_Hour_Data/AQS_hourly_data_2011_LatLon.csv"
    comparison_file = r"/backupdata/data_EPA/EQUATES/2011_Hour_Data/comparison_missing_extra_sites.csv"

    # 读取监测文件并过滤 O3 == -999 后的站点 ID
    df_monitor = pd.read_csv(monitor_file, dtype={'site_id': str})
    df_monitor_filtered = df_monitor[df_monitor['O3'] != -999]
    filtered_monitor_site_ids = df_monitor_filtered['site_id'].unique()

    # 读取站点文件并获取唯一站点 ID
    df_station = pd.read_csv(station_file, header=None, usecols=[0], names=['Site'], dtype={'Site': str})
    filtered_station_site_ids = df_station['Site'].unique()

    # 打印去除 O3 == -999 后站点数量
    print(f"去除 O3 == -999 后，监测文件中的唯一站点数量：{len(filtered_monitor_site_ids)}")
    print(f"站点文件中的唯一站点数量：{len(filtered_station_site_ids)}")

    # 找出站点文件中存在但监测文件中没有的站点
    missing_in_monitor_file = set(filtered_station_site_ids) - set(filtered_monitor_site_ids)
    extra_in_monitor_file = set(filtered_monitor_site_ids) - set(filtered_station_site_ids)

    if missing_in_monitor_file or extra_in_monitor_file:
        print("缺失的站点：", missing_in_monitor_file)
        print("多余的站点：", extra_in_monitor_file)

        # 将缺失的站点和多余的站点保存到文件
        comparison_df = pd.DataFrame({
            'Missing Site ID in merged data': list(missing_in_monitor_file),
            'Extra Site ID in merged data': list(extra_in_monitor_file)
        })
        comparison_df.to_csv(comparison_file, index=False)
        print(f"缺失和多余的站点信息已保存到 {comparison_file}")
    else:
        print("没有缺失或多余的站点。")

    # 继续执行数据合并过程
    merge_station_data(station_file, monitor_file, output_file, comparison_file)
# import pandas as pd

# # 定义文件路径
# base_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/W126'
# file1_path = f'{base_path}/2011_Monitor_W126.csv'
# file2_path = f'{base_path}/2011_Monitor_W126_SMAT.csv'

# try:
#     # 读取 CSV 文件
#     df1 = pd.read_csv(file1_path)
#     df2 = pd.read_csv(file2_path)

#     # 检查两个 DataFrame 中是否都存在 'Site' 列
#     if 'Site' in df1.columns and 'Site' in df2.columns:
#         # 提取 Site 列相同的数据
#         common_sites = df2['Site'].unique()
#         result_df = df1[df1['Site'].isin(common_sites)]

#         # 找出在 df2 中存在但在 df1 中不存在的站点
#         missing_sites = df2[~df2['Site'].isin(df1['Site'])]

#         # 确保两个 DataFrame 列名一致，不一致的列用 NaN 填充
#         all_columns = set(df1.columns).union(set(df2.columns))
#         df1 = df1.reindex(columns=all_columns)
#         df2 = df2.reindex(columns=all_columns)

#         # 将缺失站点的数据添加到结果中
#         result_df = pd.concat([result_df, missing_sites], ignore_index=True)

#         # 检查结果数据表中 O3 列是否存在缺失值
#         if 'O3' in result_df.columns and result_df['O3'].hasnans:
#             # 找出 O3 列存在缺失值的站点
#             missing_O3_sites = result_df[result_df['O3'].isna()]['Site'].unique()
#             print("O3 列存在缺失值的站点有:", missing_O3_sites)

#             # 遍历结果数据表中 O3 列的缺失值
#             for index, row in result_df[result_df['O3'].isna()].iterrows():
#                 site = row['Site']
#                 # 从 2011_Monitor_W126_SMAT.csv 中选取相同 Site 的 O3 值
#                 replacement = df2[(df2['Site'] == site) & (~df2['O3'].isna())]['O3'].values
#                 if replacement.size > 0:
#                     result_df.at[index, 'O3'] = replacement[0]

#         # 保存为新的数据表
#         new_file_path = f'{base_path}/2011_Monitor_W126_SameSiteWithSMATAddMissing.csv'
#         result_df.to_csv(new_file_path, index=False)
#         print(f"数据已成功保存到 {new_file_path}")
#     else:
#         print("其中一个文件中不存在 'Site' 列。")

# except FileNotFoundError:
#     print("错误：文件未找到，请检查文件路径是否正确。")
# except Exception as e:
#     print(f"发生未知错误：{e}")
    
    
    
    
# import pandas as pd

# # 定义文件路径
# base_path_1 = '/DeepLearning/mnt/shixiansheng/data_fusion/output/W126'
# base_path_2 = '/backupdata/data_EPA/aq_obs/routine/2011'
# file1_path = f'{base_path_1}/2011_Monitor_W126_NotSameSiteWithSMAT.csv'
# file2_path = f'{base_path_2}/AQS_hourly_data_2011_LatLon.csv'

# try:
#     # 读取 CSV 文件
#     df1 = pd.read_csv(file1_path)
#     df2 = pd.read_csv(file2_path)

#     # 检查列是否存在
#     if 'Site' in df1.columns and 'site_id' in df2.columns:
#         # 获取第一个文件中的所有 Site
#         sites_in_df1 = df1['Site'].unique()

#         # 找出在第二个文件中存在的 Site
#         existing_sites = []
#         non_existing_sites = []

#         for site in sites_in_df1:
#             if site in df2['site_id'].values:
#                 existing_sites.append(site)
#             else:
#                 non_existing_sites.append(site)

#         # 打印结果
#         print("存在的 Site:")
#         print(existing_sites)
#         print("不存在的 Site:")
#         print(non_existing_sites)
#     else:
#         print("文件中不存在所需的列，请检查列名。")

# except FileNotFoundError:
#     print("错误：文件未找到，请检查文件路径是否正确。")
# except Exception as e:
#     print(f"发生未知错误：{e}")
    
    
    