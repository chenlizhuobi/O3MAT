import pyrsig
import pyproj
import nna_methods
import os
from tqdm.auto import tqdm
import pandas as pd
import time
import numpy as np
from esil.date_helper import timer_decorator, get_day_of_year
# for show maps
from esil.rsm_helper.model_property import model_attribute
from esil.map_helper import get_multiple_data, show_maps
import cmaps
import xarray as xr
import multiprocessing  # 用于获取CPU核心数

cmap_conc = cmaps.WhiteBlueGreenYellowRed
cmap_delta = cmaps.ViBlGrWhYeOrRe


@timer_decorator
def start_daily_data_fusion(model_file, monitor_file, file_path, monitor_pollutant="ozone", model_pollutant="O3"):
    """
    @param {string} model_file: 模型文件，必须有维度：Time, Layer, ROW, COL，以及变量：O3_MDA8
    @param {string} monitor_file: 监测文件，必须有列：Site, POC, Date, Lat, Lon, Conc
    @param {string} file_path: 输出文件路径
    @param {string} monitor_pollutant: 监测文件中的污染物，默认是 ozone
    @param {string} model_pollutant: 模型文件中的污染物，默认是 O3
    """
    ds_model = pyrsig.open_ioapi(model_file)
    proj = pyproj.Proj(ds_model.crs_proj4)
    df_obs = pd.read_csv(monitor_file)
    nn = nna_methods.NNA(method="voronoi", k=30)
    df_all_daily_prediction = None

    df_obs_grouped = (
        df_obs.groupby(["Site", "Date"])
        .aggregate({"Conc": "mean", "Lat": "mean", "Lon": "mean"})
        .sort_values(by="Date")
    ).reset_index()
    dates = df_obs_grouped["Date"].unique()

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
            nn.fit(
                df_daily_obs[["x", "y"]],
                df_daily_obs[[monitor_pollutant, "mod", "bias", "r_n"]]
            )

            zdf = nn.predict(df_prediction[["COL", "ROW"]].values)
            df_prediction["vna_ozone"], df_prediction["vna_mod"], df_prediction["vna_bias"], df_prediction["vna_r_n"] = zdf.T

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
    save_daily_data_fusion_to_metrics(file_path, save_path, project_name)
    print(f"Data Fusion for all dates is done, the results are saved to {file_path}")


def start_period_averaged_data_fusion(model_file, monitor_file, file_path, dict_period, monitor_pollutant="ozone",
                                      model_pollutant="O3"):
    """
    @param {string} model_file: 模型文件，必须有维度：Time, Layer, ROW, COL，以及变量：O3_MDA8
    @param {string} monitor_file: 监测文件，必须有列：Site, POC, Date, Lat, Lon, Conc
    @param {string} file_path: 输出文件路径
    @param {dict} dict_period: 每个数据融合的时间段，键是时间段名称，值是时间段的开始和结束日期列表
    """
    ds_model = pyrsig.open_ioapi(model_file)
    proj = pyproj.Proj(ds_model.crs_proj4)
    df_obs = pd.read_csv(monitor_file)
    #控制变量
    # df_obs = df_obs[df_obs['Site'] != 60650008]
    nn = nna_methods.NNA(method="voronoi", k=30)
    df_all_daily_prediction = None
    df_obs["Date"] = pd.to_datetime(df_obs["Date"])
    
    with tqdm(dict_period.items()) as pbar:
        for peroid_name, peroid in pbar:
            start_date, end_date = peroid[0], peroid[1]
            pbar.set_description(f"Data Fusion for {peroid_name}...")

            if start_date > end_date:
                df_filtered_obs = df_obs[(df_obs["Date"] >= start_date) | (df_obs["Date"] <= end_date)]
            else:
                df_filtered_obs = df_obs[(df_obs["Date"] >= start_date) & (df_obs["Date"] <= end_date)]

            df_avg_obs = (
                df_filtered_obs.groupby(["Site"]).aggregate(
                    {"Conc": "mean", "Lat": "mean", "Lon": "mean"}
                )
            ).reset_index()

            start_time = time.time()

            if isinstance(ds_model["TSTEP"].values[0], np.int64):
                start_time_index = get_day_of_year(start_date) - 1
                end_time_index = get_day_of_year(end_date) - 1
                if start_date > end_date:
                    ds_avg_model = ds_model.sel(
                        TSTEP=list(range(start_time_index, len(ds_model["TSTEP"]))) + list(range(0, end_time_index + 1))
                    ).mean(dim="TSTEP")
                else:
                    ds_avg_model = ds_model.sel(
                        TSTEP=slice(start_time_index, end_time_index)
                    ).mean(dim="TSTEP")
            else:
                if start_date > end_date:
                    ds_avg_model = ds_model.sel(
                        TSTEP=list(pd.date_range(start=start_date, end='2011-12-31')) + list(
                            pd.date_range(start='2011-01-01', end=end_date))
                    ).mean(dim="TSTEP")
                else:
                    ds_avg_model = ds_model.sel(TSTEP=slice(start_date, end_date)).mean(dim="TSTEP")

            df_avg_obs["x"], df_avg_obs["y"] = proj(df_avg_obs["Lon"], df_avg_obs["Lat"])
            df_avg_obs["mod"] = ds_avg_model[model_pollutant][0].sel(
                ROW=df_avg_obs["y"].to_xarray(),
                COL=df_avg_obs["x"].to_xarray(),
                method="nearest"
            )
            df_avg_obs["bias"] = df_avg_obs["mod"] - df_avg_obs["Conc"]
            df_avg_obs["r_n"] = df_avg_obs["Conc"] / df_avg_obs["mod"]
            df_prediction = ds_avg_model[["ROW", "COL"]].to_dataframe().reset_index()
            nn.fit(
                df_avg_obs[["x", "y"]],
                df_avg_obs[[monitor_pollutant, "mod", "bias", "r_n"]]
            )
            njobs = multiprocessing.cpu_count()
            zdf = nn.predict(df_prediction[["COL", "ROW"]].values,njobs=njobs)
            df_prediction["vna_ozone"], df_prediction["vna_mod"], df_prediction["vna_bias"], df_prediction["vna_r_n"] = zdf.T

            df_fusion = df_prediction.set_index(["ROW", "COL"]).to_xarray()
            df_fusion["avna_ozone"] = ds_avg_model[model_pollutant][0].values - df_fusion["vna_bias"]
            reshaped_vna_r_n = df_prediction["vna_r_n"].values.reshape(ds_avg_model[model_pollutant][0].shape)
            df_fusion["evna_ozone"] = (("ROW", "COL"), ds_avg_model[model_pollutant][0].values * reshaped_vna_r_n)
            df_fusion = df_fusion.to_dataframe().reset_index()
            df_fusion["model"] = ds_avg_model[model_pollutant][0].values.flatten()
            df_fusion["Period"] = peroid_name
            df_fusion["COL"] = (df_fusion["COL"] + 0.5).astype(int)
            df_fusion["ROW"] = (df_fusion["ROW"] + 0.5).astype(int)

            if df_all_daily_prediction is None:
                df_all_daily_prediction = df_fusion
            else:
                df_all_daily_prediction = pd.concat([df_all_daily_prediction, df_fusion])

            end_time = time.time()
            duration = end_time - start_time
            print(f"Data Fusion for {peroid_name} took {duration:.2f} seconds")

    df_all_daily_prediction.to_csv(file_path, index=False)
    print(f"Data Fusion for all dates is done, the results are saved to {file_path}")

def start_top_Av_DF(model_file, monitor_file, file_path, monitor_pollutant="ozone", model_pollutant="O3_MDA8"):
    start_time = time.time()
    ds_model = pyrsig.open_ioapi(model_file)
    proj = pyproj.Proj(ds_model.crs_proj4)
    df_obs = pd.read_csv(monitor_file)
    # Filter out rows where 'Site' is 60650008
    df_obs = df_obs[df_obs['Site'] != 60650008]
    nn = nna_methods.NNA(method="voronoi", k=30)
    df_all_daily_prediction = None
    df_obs["Date"] = pd.to_datetime(df_obs["Date"])

    def top_n_average(series, n, is_98th=False):
        if is_98th:
            return series.nlargest(n).min()
        return series.nlargest(n).mean()

    # 处理 top-10
    df_avg_obs_top10 = (
        df_obs.groupby(["Site"]).aggregate(
            {"Conc": lambda x: top_n_average(x, 10), "Lat": "mean", "Lon": "mean"}
        )
    ).reset_index()

    # 处理 98th（监测站数据）
    df_avg_obs_98th = (
        df_obs.groupby(["Site"]).aggregate(
            {"Conc": lambda x: top_n_average(x, 7, is_98th=True), "Lat": "mean", "Lon": "mean"}
        )
    ).reset_index()

    def get_top_avg(model, model_pollutant, n, is_98th=False):
        """
        计算每个网格位置上污染物浓度降序排序后的前 n 个值的平均值。
        若 is_98th 为 True 且处理模型数据，计算 98 分位数。
        """
        pollutant_data = model[model_pollutant]

        def sort_and_take_top(arr, n, is_98th):
            if is_98th:
                return np.quantile(arr, 0.98)
            sorted_arr = np.sort(arr)[::-1]
            return sorted_arr[:n].mean()

        avg_data = xr.apply_ufunc(
            sort_and_take_top,
            pollutant_data,
            input_core_dims=[['TSTEP']],
            kwargs={'n': n, 'is_98th': is_98th},
            vectorize=True
        )

        return avg_data

    # 调用 get_top_avg 来获取 top-10 和 98th 值
    df_avg_model_top10 = get_top_avg(ds_model, model_pollutant, 10)
    df_avg_model_98th = get_top_avg(ds_model, model_pollutant, 7, is_98th=True)

    for period_name, df_avg_obs, ds_avg_model in [("top-10", df_avg_obs_top10, df_avg_model_top10),
                                                  ("98th", df_avg_obs_98th, df_avg_model_98th)]:
        print(f"Data Fusion for {period_name}...")

        df_avg_obs["x"], df_avg_obs["y"] = proj(df_avg_obs["Lon"], df_avg_obs["Lat"])

        # 确保正确访问模型数据并处理索引
        df_avg_obs["mod"] = ds_avg_model.sel(
            LAY=0,
            ROW=df_avg_obs["y"].to_xarray(),
            COL=df_avg_obs["x"].to_xarray(),
            method="nearest"
        )
        ds_avg_model_dataset = ds_avg_model.to_dataset(name='O3_MDA8')

        # 如果你想同时处理坐标，可以添加坐标信息作为数据变量
        ds_avg_model_dataset['ROW'] = ds_avg_model_dataset.ROW
        ds_avg_model_dataset['COL'] = ds_avg_model_dataset.COL

        # 现在可以像下面这样选取多个数据变量
        selected_ds = ds_avg_model_dataset[['O3_MDA8', 'ROW', 'COL']]
        print(selected_ds)

        df_avg_obs["bias"] = df_avg_obs["mod"] - df_avg_obs["Conc"]
        df_avg_obs["r_n"] = df_avg_obs["Conc"] / df_avg_obs["mod"]

        df_prediction = selected_ds[["ROW", "COL"]].to_dataframe().reset_index()
        nn.fit(
            df_avg_obs[["x", "y"]],
            df_avg_obs[[monitor_pollutant, "mod", "bias", "r_n"]]
        )

        zdf = nn.predict(df_prediction[["COL", "ROW"]].values)
        df_prediction["vna_ozone"], df_prediction["vna_mod"], df_prediction["vna_bias"], df_prediction["vna_r_n"] = zdf.T

        df_fusion = df_prediction.set_index(["ROW", "COL"]).to_xarray()
        # 从 selected_ds 中选取 O3_MDA8 数据变量并获取其值
        o3_mda8_values = selected_ds['O3_MDA8'].values
        # 去除 LAY 维度（因为 LAY 只有一个元素）
        o3_mda8_values_2d = o3_mda8_values.squeeze()

        # 计算 avna_ozone 的值
        avna_ozone_values = o3_mda8_values_2d - df_fusion["vna_bias"].values
        # 明确指定维度名称
        df_fusion["avna_ozone"] = (('ROW', 'COL'), avna_ozone_values)

        reshaped_vna_r_n = df_prediction["vna_r_n"].values.reshape(o3_mda8_values_2d.shape)
        df_fusion["evna_ozone"] = (('ROW', 'COL'), o3_mda8_values_2d * reshaped_vna_r_n)
        df_fusion = df_fusion.to_dataframe().reset_index()
        df_fusion["model"] = o3_mda8_values_2d.flatten()
        df_fusion["Period"] = period_name
        df_fusion["COL"] = (df_fusion["COL"] + 0.5).astype(int)
        df_fusion["ROW"] = (df_fusion["ROW"] + 0.5).astype(int)

        if df_all_daily_prediction is None:
            df_all_daily_prediction = df_fusion
        else:
            df_all_daily_prediction = pd.concat([df_all_daily_prediction, df_fusion])

    end_time = time.time()
    duration = end_time - start_time
    print(f"Data Fusion for all top values took {duration:.2f} seconds")

    if df_all_daily_prediction is not None:
        df_all_daily_prediction.to_csv(file_path, index=False)
        print(f"Data Fusion for all top values is done, the results are saved to {file_path}")
    else:
        print("没有有效的融合数据，未保存结果。")


@timer_decorator
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

    # 98th percentile of MDA8 ozone concentration
    df_data_98th_percentile = df_data.groupby(["ROW", "COL"]).agg(
        {'vna_ozone': lambda x: x.quantile(0.98),
         'evna_ozone': lambda x: x.quantile(0.98),
         'avna_ozone': lambda x: x.quantile(0.98),
         'model': lambda x: x.quantile(0.98)}
    ).reset_index()
    df_data_98th_percentile["Period"] = f"98th"
    all_metrics.append(df_data_98th_percentile)

    # top-10 average of MDA8 ozone days
    def top_10_average(series):
        return series.nlargest(10).mean()

    df_data_top_10_avg = df_data.groupby(["ROW", "COL"]).agg(
        {'vna_ozone': top_10_average,
         'evna_ozone': top_10_average,
         'avna_ozone': top_10_average,
         'model': top_10_average}
    ).reset_index()
    df_data_top_10_avg["Period"] = f"top-10"
    all_metrics.append(df_data_top_10_avg)

    # Annual average of MDA8
    df_data_annual_avg = df_data.groupby(["ROW", "COL", 'Year']).agg(
        {'vna_ozone': 'mean',
         'evna_ozone': 'mean',
         'avna_ozone': 'mean',
         'model': 'mean'}
    ).reset_index()
    df_data_annual_avg["Period"] = f"Annual"
    all_metrics.append(df_data_annual_avg)

    # Summer season average (Apr-Sep) of MDA8
    summer_months = [4, 5, 6, 7, 8, 9]
    df_data_summer = df_data[df_data['Month'].isin(summer_months)]
    df_data_summer_avg = df_data_summer.groupby(["ROW", "COL"]).agg(
        {'vna_ozone': 'mean',
         'evna_ozone': 'mean',
         'avna_ozone': 'mean',
         'model': 'mean'}
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
            {'vna_ozone': 'mean',
             'evna_ozone': 'mean',
             'avna_ozone': 'mean',
             'model': 'mean'}
        ).reset_index()
        df_data_season_avg["Period"] = f"{season}"
        all_metrics.append(df_data_season_avg)

    # 合并所有指标到一个 DataFrame
    final_df = pd.concat(all_metrics, ignore_index=True)

    # 保存为一个 CSV 文件
    output_file = os.path.join(save_path, f"{project_name}") 
    final_df.to_csv(output_file, index=False)
    output_file_list.append(output_file)

    return output_file_list


if __name__ == "__main__":
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/Data_OTHER/2011_Other"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_file_name = "2011_HarvardBased_AtFdaily.csv"
    output_file_path = os.path.join(save_path, output_file_name)
    output_file_name_season = "2011_HarvardBased_AtF.csv"

    model_file = r"/backupdata/data_EPA/Harvard/unzipped_tifs/Harvard_O3MDA8_Regridded_grid_center_2011_12km.nc"
    # model_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/HR2DAY_LST_ACONC_v532_cb6r3_ae7_aq_WR413_MYR_STAGE_2011_12US1_2011.nc"
    # monitor_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv"
    monitor_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv"

    # 测试 先融合后均值
# # 测试 先融合后均值
#     daily_output_path = os.path.join(save_path, "BarronScriptHarvard_ALL_2011_FtAdaily.csv")
#     start_daily_data_fusion(
#         model_file,
#         monitor_file,
#         daily_output_path,
#         monitor_pollutant="Conc",
#         model_pollutant="MDA8_O3"
#     )
#     daily_output_path = os.path.join(save_path, "BarronScript_ALL_2011_FtAdaily.csv")
#     # 动态生成先融合后均值的指标文件名称
#     daily_index_file_name = daily_output_path.replace("FtAdaily.csv", "FtAIndex.csv")
#     daily_data_fusion_file = daily_output_path
#     daily_data=pd.read_csv(daily_data_fusion_file)
#     # 将先融合后均值的结果处理为相应的指标
#     daily_file_list = save_daily_data_fusion_to_metrics(daily_data, save_path, project_name="BarronHarvard_ALL_2011_FtAIndex")


    # 测试 先均值后融合
    seasonal_output_path = os.path.join(save_path, "2011_HarvardBased_AtF.csv")
    start_period_averaged_data_fusion(
        model_file,
        monitor_file,
        seasonal_output_path,
        monitor_pollutant="Conc",
        model_pollutant="MDA8_O3",
        dict_period={
            "DJF_2011": ["2011-12-01", "2011-02-28"],
            "MAM_2011": ["2011-03-01", "2011-05-31"],
            "JJA_2011": ["2011-06-01", "2011-08-31"],
            "SON_2011": ["2011-09-01", "2011-11-30"],
            "Annual_2011": ["2011-01-01", "2011-12-31"],
            "Apr-Sep_2011": ["2011-04-01", "2011-09-30"]
        }
    )
    print("Done!")

    seasonal_output_path = os.path.join(save_path, "2011_HarvardBased_AtF.csv")
    start_top_Av_DF(
        model_file,
        monitor_file,
        seasonal_output_path,
        monitor_pollutant="Conc",
        model_pollutant="MDA8_O3"
    )