import pyrsig
import pyproj
import nna_methods  # 引入并行版本的NNA类
import os
from tqdm.auto import tqdm
import pandas as pd
import time
import numpy as np
from esil.date_helper import timer_decorator
import multiprocessing  # 用于获取CPU核心数
from scipy.spatial import KDTree


def fill_missing_timezone_info(df_obs, df_gmt_offset):
    # 合并两个 DataFrame
    df_merged = pd.merge(df_obs, df_gmt_offset, on='site_id', how='left')
    # 找出缺失时区信息的行
    missing_rows = df_merged[df_merged['gmt_offset'].isna()]
    if len(missing_rows) == 0:
        print("所有的 site_id 在 df_gmt_offset 中都能找到对应信息，无需进行额外处理。")
        df_merged = df_merged.rename(columns={'Lat_x': 'Lat', 'Lon_x': 'Lon'})
        df_merged = df_merged.drop(columns=['Lat_y', 'Lon_y'])
        print(df_merged)
        return df_merged

    print(f"存在 {len(missing_rows)} 个 site_id 在 df_gmt_offset 中找不到对应信息，将使用 KDTree 查找最近点的时区信息。")
    # 输出缺失行的 id 和经纬度
    missing_info = missing_rows[['site_id', 'Lat_x', 'Lon_x']]
    print("缺失时区信息的站点 id 和经纬度：")
    print(missing_info)

    # 构建 KDTree，使用合并后 DataFrame 中有时区信息的行的经纬度
    valid_rows = df_merged[~df_merged['gmt_offset'].isna()]
    tree = KDTree(valid_rows[['Lat_y', 'Lon_y']])
    for idx, row in missing_rows.iterrows():
        missing_lat = row['Lat_x']
        missing_lon = row['Lon_x']
        # 查询最近点的索引
        _, index = tree.query([missing_lat, missing_lon])
        # 确保 index 是整数
        if isinstance(index, np.ndarray):
            index = index[0]
        # 获取最近点的时区信息
        nearest_gmt_offset = valid_rows.iloc[index]['gmt_offset']
        nearest_epa_region = valid_rows.iloc[index]['epa_region']
        # 将最近点的时区信息填充到合并后的 DataFrame 中
        df_merged.loc[idx, 'gmt_offset'] = nearest_gmt_offset
        df_merged.loc[idx, 'epa_region'] = nearest_epa_region
    # 只保留一组经纬度信息
    df_merged = df_merged.rename(columns={'Lat_x': 'Lat', 'Lon_x': 'Lon'})
    df_merged = df_merged.drop(columns=['Lat_y', 'Lon_y'])

    # 筛选出之前缺失时区信息的行现在的时区信息
    filled_missing = df_merged.loc[missing_rows.index, ['gmt_offset', 'epa_region']]
    # 打印唯一值
    print("缺失点获取到的时区信息（唯一值）：")
    print(filled_missing.drop_duplicates())

    print(df_merged)
    return df_merged


@timer_decorator
def start_hourly_data_fusion(model_files, monitor_file_template, region_table_file, time_region_file, file_path,
                             monitor_pollutant="O3", model_pollutant="O3", start_date=None, end_date=None):
    """
    @param {list} model_files: 包含12个月模型数据的文件路径列表，每个文件对应一个月
    @param {string} monitor_file_template: 监测文件模板，使用 {year} 占位符
    @param {string} region_table_file: 包含 Is 列的数据表文件路径
    @param {string} time_region_file: 包含站点ID和TimeRegion两列的文件路径
    @param {string} file_path: 输出文件路径
    @param {string} monitor_pollutant: 监测文件中的污染物，默认是 ozone
    @param {string} model_pollutant: 模型文件中的污染物，默认是 O3
    @param {string} start_date: 开始日期，格式为 'YYYY-MM-DD HH:00'
    @param {string} end_date: 结束日期，格式为 'YYYY-MM-DD HH:00'
    """
    if start_date and end_date:
        start = pd.to_datetime(start_date.replace('/', '-'))
        end = pd.to_datetime(end_date.replace('/', '-'))
        filter_start = start - pd.Timedelta(hours=8)
        filter_end = end - pd.Timedelta(hours=4)

        # 如果是UTC第一天，左偏移读去年监测数据，同时考虑end year第一天数据
        years_to_read = [start.year]
        if start.month == 1 and start.day == 1:
            years_to_read.append(start.year - 1)
        if end.month == 1 and end.day == 1:
            years_to_read.append(end.year)

        df_obs_list = []
        for year in years_to_read:
            monitor_file = monitor_file_template.format(year=year)
            df = pd.read_csv(monitor_file)
            df['dateon'] = pd.to_datetime(df['dateon'])
            df = df[(df['dateon'] >= filter_start) & (df['dateon'] <= filter_end)]
            df_obs_list.append(df)
        df_obs = pd.concat(df_obs_list, ignore_index=True)

    else:
        # 如果没有指定日期范围，按原逻辑读取数据
        monitor_file = monitor_file_template.format(year=2019)
        df_obs = pd.read_csv(monitor_file)
        df_obs['dateon'] = pd.to_datetime(df_obs['dateon'])

    # 读取时区偏移量信息,标准时间-5~-8
    df_gmt_offset = pd.read_csv('output/Region/MonitorsTimeRegion_Filter_ST.csv')

    # 调用函数填充缺失的时区信息，一般不会缺失
    df_obs = fill_missing_timezone_info(df_obs, df_gmt_offset)

    # 监测点的ST转为对应UTC，比如最东部UTC=ST - (-5)
    df_obs['utc_dateon'] = df_obs.apply(
        lambda row: row['dateon'] - pd.Timedelta(hours=row['gmt_offset']), axis=1
    )

    # 处理日期范围
    if start_date and end_date:
        start = pd.to_datetime(start_date.replace('/', '-'))  # 修改日期格式
        end = pd.to_datetime(end_date.replace('/', '-'))  # 修改日期格式
        df_obs = df_obs[(df_obs['utc_dateon'] >= start) & (df_obs['utc_dateon'] <= end)]

    # 按站点和日期聚合
    df_obs_grouped = (
        df_obs.groupby(["site_id", "utc_dateon"])
        .agg({"O3": "mean", "Lat": "mean", "Lon": "mean"})
        .reset_index()
    )
    dates = df_obs_grouped["utc_dateon"].unique()

    # sort dates,如果不sort的话当日期范围跨过3月时就会出现从3月开始的现象
    dates = np.sort(dates)

    # 读取包含 Is 列的数据表
    region_df = pd.read_csv(region_table_file)

    # 筛选出 Is 列值为 1 的行
    us_region_df = region_df[region_df['Is'] == 1]
    # 进行偏移以确保正确的坐标,网格中心点
    us_region_df[['COL', 'ROW']] = us_region_df[['COL', 'ROW']] - 0.5
    # 获取模型的行列坐标
    us_region_row_col = us_region_df[['COL', 'ROW']].values

    # 一次性读取所有模型文件
    ds_models = [pyrsig.open_ioapi(model_file) for model_file in model_files]

    # 使用NNA进行站点数据的匹配
    nn = nna_methods.NNA(method="voronoi", k=30)  # 使用并行版本的NNA
    all_results = []

    with tqdm(dates) as pbar:
        for date in pbar:
            pbar.set_description(f"Data Fusion for {date}...")
            start_time = time.time()

            # 将numpy.datetime64转换为datetime对象
            date = pd.Timestamp(date).to_pydatetime()

            # Model的时间本身就是UTC
            adjusted_date = date
            print(f"UTC: {adjusted_date}\n")

            # 获取当天的监测数据
            df_daily_obs = df_obs_grouped[df_obs_grouped["utc_dateon"] == date].copy()
            year = adjusted_date.year
            month = adjusted_date.month

            # 根据 UTC 时间的年份和月份选择模型文件,适用全年
            file_index = (year - 2019) * 12 + month - 1
            if file_index < 0 or file_index >= len(ds_models):
                print(f"警告：没有对应的模型文件，UTC 日期: {adjusted_date}，跳过处理。")
                continue

            # 获取对应月份的模型数据
            ds_model = ds_models[file_index]
            proj = pyproj.Proj(ds_model.crs_proj4)

            # 将经纬度转换为模型的x, y坐标
            df_daily_obs["x"], df_daily_obs["y"] = proj(df_daily_obs["Lon"], df_daily_obs["Lat"])

            # 获取当天的模型数据
            tstep_value = pd.Timestamp(f"{adjusted_date.year}-{month:02d}-{adjusted_date.day:02d} {adjusted_date.hour:02d}:00:00")
            try:
                ds_hourly_model = ds_model.sel(TSTEP=tstep_value)
            except KeyError:
                print(f"警告：对应的 UTC 日期 {adjusted_date} 的模型数据缺失，跳过处理。")
                continue

            # 选择模型数据，并计算偏差
            df_daily_obs["mod"] = ds_hourly_model[model_pollutant][0].sel(
                ROW=df_daily_obs["y"].to_xarray(),
                COL=df_daily_obs["x"].to_xarray(),
                method="nearest"
            )
            df_daily_obs["bias"] = df_daily_obs["mod"] - df_daily_obs["O3"]
            df_daily_obs["r_n"] = df_daily_obs["O3"] / df_daily_obs["mod"]
            # 若 r_n 超过3，则将其设为3
            df_daily_obs.loc[df_daily_obs["r_n"] > 3, "r_n"] = 3

            df_prediction = ds_hourly_model[["ROW", "COL"]].to_dataframe().reset_index()

            # 训练NNA模型
            nn.fit(
                df_daily_obs[["x", "y"]],
                df_daily_obs[[monitor_pollutant, "mod", "bias", "r_n"]]
            )

            # 并行计算
            njobs = multiprocessing.cpu_count()  # 使用所有CPU核心进行并行计算
            zdf = nn.predict(us_region_row_col, njobs=njobs)

            # 创建一个全为NaN的DataFrame来存储预测结果
            result_df = pd.DataFrame(np.nan, index=df_prediction.index, columns=["vna_ozone", "vna_mod", "vna_bias", "vna_r_n"])

            # 将美国区域的预测结果填充到对应的行
            result_df.loc[us_region_df.index] = zdf

            df_prediction = pd.concat([df_prediction, result_df], axis=1)

            df_fusion = df_prediction.set_index(["ROW", "COL"]).to_xarray()
            df_fusion["avna_ozone"] = ds_hourly_model[model_pollutant][0].values - df_fusion["vna_bias"]
            reshaped_vna_r_n = df_prediction["vna_r_n"].values.reshape(ds_hourly_model[model_pollutant][0].shape)
            df_fusion["evna_ozone"] = (("ROW", "COL"), ds_hourly_model[model_pollutant][0].values * reshaped_vna_r_n)
            df_fusion = df_fusion.to_dataframe().reset_index()
            df_fusion["model"] = ds_hourly_model[model_pollutant][0].values.flatten()
            # 修改日期格式为带小时的形式
            df_fusion["TSTEP"] = adjusted_date.strftime('%Y-%m-%d %H:%M:%S')  # Model的UTC 时间
            df_fusion["Timestamp"] = date.strftime('%Y-%m-%d %H:%M:%S')  # Monitor调整后的UTC时间
            df_fusion["COL"] = (df_fusion["COL"] + 0.5).astype(int)
            df_fusion["ROW"] = (df_fusion["ROW"] + 0.5).astype(int)

            all_results.append(df_fusion)

            end_time = time.time()
            duration = end_time - start_time
            print(f"Data Fusion for {date} took {duration:.2f} seconds")

    # 一次性合并所有结果
    df_all_hourly_prediction = pd.concat(all_results, ignore_index=True)

    df_all_hourly_prediction.to_csv(file_path, index=False)
    print(f"Data Fusion for all dates is done, the results are saved to {file_path}")
    return df_all_hourly_prediction


# 在 main 函数中调用
if __name__ == "__main__":
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_files = [
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201901.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201902.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201903.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201904.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201905.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201906.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201907.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201908.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201909.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201910.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201911.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201912.nc",
    ]

    # 使用占位符 {year} 表示年份
    monitor_file_template = r"/backupdata/data_EPA/aq_obs/routine/{year}/AQS_hourly_data_{year}_LatLon.csv"
    region_table_file = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/Region_CONUSHarvard.csv"
    time_region_file = r"output/Region/MonitorsTimeRegion_Filter_ST.csv"  # 请替换为实际的TimeRegion文件路径

    # 指定日期范围
    start_date = '2019/03/01 00:00'
    end_date = '2019/11/01 08:00'

    daily_output_path = os.path.join(save_path, "2019_HourlyData_Limit.csv")
    start_hourly_data_fusion(
        model_files,
        monitor_file_template,
        region_table_file,
        time_region_file,
        daily_output_path,
        monitor_pollutant="O3",
        model_pollutant="O3",
        start_date=start_date,
        end_date=end_date,
    )
    print("Done!")
    