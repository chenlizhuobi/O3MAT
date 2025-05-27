import pyrsig
import pyproj
import nna_methods  # 引入并行版本的NNA类
import os
from tqdm.auto import tqdm
import pandas as pd
import time
import numpy as np
from esil.date_helper import timer_decorator, get_day_of_year
from esil.map_helper import show_maps
import cmaps
import xarray as xr
import multiprocessing  # 用于获取CPU核心数


cmap_conc = cmaps.WhiteBlueGreenYellowRed
cmap_delta = cmaps.ViBlGrWhYeOrRe


@timer_decorator
def start_daily_data_fusion(model_file, monitor_file, region_table_file, file_path, monitor_pollutant="ozone",
                            model_pollutant="O3", start_date=None, end_date=None):
    """
    @param {string} model_file: 模型文件，必须有维度：Time, Layer, ROW, COL，以及变量：O3_MDA8
    @param {string} monitor_file: 监测文件，必须有列：Site, POC, Date, Lat, Lon, Conc
    @param {string} region_table_file: 包含 Is 列的数据表文件路径
    @param {string} file_path: 输出文件路径
    @param {string} monitor_pollutant: 监测文件中的污染物，默认是 ozone
    @param {string} model_pollutant: 模型文件中的污染物，默认是 O3
    @param {string} start_date: 开始日期，格式为 'YYYY-MM-DD'
    @param {string} end_date: 结束日期，格式为 'YYYY-MM-DD'
    @param {string} region_df: 输入融合的区域文件, ROW COL Is三列表示需要融合的区域
    """
    ds_model = pyrsig.open_ioapi(model_file)
    proj = pyproj.Proj(ds_model.crs_proj4)
    df_obs = pd.read_csv(monitor_file)
    nn = nna_methods.NNA(method="voronoi", k=30)  # 使用并行版本的NNA
    df_all_daily_prediction = None

    df_obs_grouped = (
        df_obs.groupby(["Site", "Date"])
       .aggregate({"Conc": "mean", "Lat": "mean", "Lon": "mean"})
       .sort_values(by="Date")
    ).reset_index()
    if start_date and end_date:
        df_obs_grouped = df_obs_grouped[(df_obs_grouped["Date"] >= start_date) & (df_obs_grouped["Date"] <= end_date)]
    dates = df_obs_grouped["Date"].unique()

    # 读取包含 Is 列的数据表
    region_df = pd.read_csv(region_table_file)

    # 筛选出 Is 列值为 1 的行
    us_region_df = region_df[region_df['Is'] == 1]
    #-0.5确保后面的计算是正确的
    us_region_df[['COL', 'ROW']] = us_region_df[['COL', 'ROW']] - 0.5
    # 输入为x,y形式，对于model的col和row
    us_region_row_col = us_region_df[['COL', 'ROW']].values

    with tqdm(dates) as pbar:
        for date in pbar:
            pbar.set_description(f"Data Fusion for {date}...")
            start_time = time.time()

            df_daily_obs = df_obs_grouped[df_obs_grouped["Date"] == date].copy()
            if isinstance(ds_model['TSTEP'].values[0], np.int64):
                timeIndex = get_day_of_year(date) - 1
                ds_daily_model = ds_model.sel(TSTEP=timeIndex)
            else:
                ds_daily_model = ds_model.sel(TSTEP=date)

            df_daily_obs["x"], df_daily_obs["y"] = proj(df_daily_obs["Lon"], df_daily_obs["Lat"])
            df_daily_obs["mod"] = ds_daily_model[model_pollutant][0].sel(
                ROW=df_daily_obs["y"].to_xarray(),
                COL=df_daily_obs["x"].to_xarray(),
                method="nearest"
            )
            df_daily_obs["bias"] = df_daily_obs["mod"] - df_daily_obs["Conc"]
            df_daily_obs["r_n"] = df_daily_obs["Conc"] / df_daily_obs["mod"]

            df_prediction = ds_daily_model[["ROW", "COL"]].to_dataframe().reset_index()
            # 通过训练使得网格点能快速搜寻临近点
            nn.fit(
                df_daily_obs[["x", "y"]],
                df_daily_obs[[monitor_pollutant, "mod", "bias", "r_n"]]
            )

            # 并行计算部分
            njobs = multiprocessing.cpu_count()  # 使用所有CPU核心进行并行计算
            zdf = nn.predict(us_region_row_col, njobs=njobs)

            # 创建一个全为 NaN 的 DataFrame 用于存储预测结果
            result_df = pd.DataFrame(np.nan, index=df_prediction.index, columns=["vna_ozone", "vna_mod", "vna_bias", "vna_r_n"])

            # 将美国区域的预测结果填充到对应的行
            result_df.loc[us_region_df.index] = zdf

            df_prediction = pd.concat([df_prediction, result_df], axis=1)

            df_fusion = df_prediction.set_index(["ROW", "COL"]).to_xarray()
            df_fusion["avna_ozone"] = ds_daily_model[model_pollutant][0].values - df_fusion["vna_bias"]
            reshaped_vna_r_n = df_prediction["vna_r_n"].values.reshape(ds_daily_model[model_pollutant][0].shape)
            df_fusion["evna_ozone"] = (("ROW", "COL"), ds_daily_model[model_pollutant][0].values * reshaped_vna_r_n)
            df_fusion = df_fusion.to_dataframe().reset_index()
            df_fusion["model"] = ds_daily_model[model_pollutant][0].values.flatten()
            df_fusion["Timestamp"] = date
            df_fusion["COL"] = (df_fusion["COL"] + 0.5).astype(int)
            df_fusion["ROW"] = (df_fusion["ROW"] + 0.5).astype(int)

            if df_all_daily_prediction is None:
                df_all_daily_prediction = df_fusion
            else:
                df_all_daily_prediction = pd.concat([df_all_daily_prediction, df_fusion])

            end_time = time.time()
            duration = end_time - start_time
            print(f"Data Fusion for {date} took {duration:.2f} seconds")

    df_all_daily_prediction.to_csv(file_path, index=False)
    project_name = os.path.basename(file_path).replace(".csv", "")
    print(f"Data Fusion for all dates is done, the results are saved to {file_path}")

    # 调用 save_daily_data_fusion_to_metrics 函数
    save_path = os.path.dirname(file_path)
    output_file_list = save_daily_data_fusion_to_metrics(df_all_daily_prediction, save_path, project_name)
    print(f"O3-related metrics files saved: {output_file_list}")
    return df_all_daily_prediction


def save_daily_data_fusion_to_metrics(df_data, save_path, project_name):
    '''
    This function converts the daily data fusion results to O3-related metrics (e.g., 98th percentile of MDA8 ozone concentration, average of top-10 MDA8 ozone days, annual average of MDA8 ozone concentration) files.
    @param {DataFrame} df_data: The DataFrame of the daily data fusion results.
    @param {str} save_path: The path to save the O3-related metrics files.
    @param {str} project_name: The name of the project.
    @return {list} output_file_list: The list of the O3-related metrics files.
    '''
    output_file_list = []

    # 提取年份和月份
    df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])
    df_data['Year'] = df_data['Timestamp'].dt.year
    df_data['Month'] = df_data['Timestamp'].dt.month

    # 初始化一个空的 DataFrame 来存储所有指标
    all_metrics = []

    # # 98th percentile of MDA8 ozone concentration
    # df_data_98th_percentile = df_data.groupby(["ROW", "COL"]).agg(
    #     {'vna_ozone': lambda x: x.quantile(0.98),
    #      'evna_ozone': lambda x: x.quantile(0.98),
    #      'avna_ozone': lambda x: x.quantile(0.98),
    #     'model': lambda x: x.quantile(0.98)}
    # ).reset_index()
    # df_data_98th_percentile["Period"] = f"98th"
    # all_metrics.append(df_data_98th_percentile)

    # top-10 average of MDA8 ozone days
    # def top_10_average(series):
    #     return series.nlargest(10).mean()

    # df_data_top_10_avg = df_data.groupby(["ROW", "COL"]).agg(
    #     {'vna_ozone': top_10_average,
    #      'evna_ozone': top_10_average,
    #      'avna_ozone': top_10_average,
    #     'model': top_10_average}
    # ).reset_index()
    # df_data_top_10_avg["Period"] = f"top-10"
    # all_metrics.append(df_data_top_10_avg)

    # Annual average of MDA8
    df_data_annual_avg = df_data.groupby(["ROW", "COL", 'Year']).agg(
        {'vna_ozone':'mean',
         'evna_ozone':'mean',
         'avna_ozone':'mean',
        'model':'mean'}
    ).reset_index()
    df_data_annual_avg["Period"] = f"Annual"
    all_metrics.append(df_data_annual_avg)

    # Summer season average (Apr-Sep) of MDA8
    summer_months = [4, 5, 6, 7, 8, 9]
    df_data_summer = df_data[df_data['Month'].isin(summer_months)]
    df_data_summer_avg = df_data_summer.groupby(["ROW", "COL"]).agg(
        {'vna_ozone':'mean',
         'evna_ozone':'mean',
         'avna_ozone':'mean',
        'model':'mean'}
    ).reset_index()
    df_data_summer_avg["Period"] = f"Apr-Sep"
    all_metrics.append(df_data_summer_avg)

    # seasonal averages（DJF, MAM, JJA, SON）of MDA8
    seasons = {
        'DJF': [12, 1, 2],  # December,January, Feburary
        'MAM': [3, 4, 5],  # April, May, June
        'JJA': [6, 7, 8],  # July, August, September
        'SON': [9, 10, 11]  # October, November, December
    }
    for season, months in seasons.items():
        df_data_season = df_data[df_data['Month'].isin(months)]
        df_data_season_avg = df_data_season.groupby(["ROW", "COL"]).agg(
            {'vna_ozone':'mean',
             'evna_ozone':'mean',
             'avna_ozone':'mean',
            'model':'mean'}
        ).reset_index()
        df_data_season_avg["Period"] = f"{season}"
        all_metrics.append(df_data_season_avg)

    # 合并所有指标到一个 DataFrame
    final_df = pd.concat(all_metrics, ignore_index=True)

    # 保存为一个 CSV 文件
    output_file = os.path.join(save_path, f"{project_name}_metrics.csv")
    final_df.to_csv(output_file, index=False)
    output_file_list.append(output_file)

    return output_file_list


# 在 main 函数中调用
if __name__ == "__main__":
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV"
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/Data_OTHER/2011_Other"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # model_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/HR2DAY_LST_ACONC_v532_cb6r3_ae7_aq_WR413_MYR_STAGE_2011_12US1_2011.nc"
    model_file = r"/backupdata/data_EPA/Harvard/unzipped_tifs/Harvard_O3MDA8_Regridded_grid_center_2011_12km.nc"
    monitor_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv"
    region_table_file = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/Region_CONUSHarvard.csv"  # 替换为实际的包含 Is 列的数据表文件路径

    # 指定日期范围
    start_date = '2011-01-01'
    end_date = '2011-12-31'

    daily_output_path = os.path.join(save_path, "2011_HarvardBased_FtA.csv")
    start_daily_data_fusion(
        model_file,
        monitor_file,
        region_table_file,
        daily_output_path,
        monitor_pollutant="Conc",
        model_pollutant="MDA8_O3",
        start_date=start_date,
        end_date=end_date
    )
    print("Done!")