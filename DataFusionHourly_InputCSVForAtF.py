import pyrsig
import pyproj
import nna_methods  # 引入并行版本的NNA类
import os
from tqdm.auto import tqdm
import pandas as pd
import time
import numpy as np
import multiprocessing  # 用于获取CPU核心数
import xarray as xr


def start_daily_data_fusion(model_file, monitor_file, region_table_file, file_path, monitor_pollutant="ozone",
                            model_pollutant="O3"):
    """
    @param {string} model_file: 模型文件，必须有列：ROW, COL, model
    @param {string} monitor_file: 监测文件，必须有列：Site, Lat, Lon, O3
    @param {string} region_table_file: 包含 Is 列的数据表文件路径
    @param {string} file_path: 输出文件路径
    @param {string} monitor_pollutant: 监测文件中的污染物，默认是 ozone
    @param {string} model_pollutant: 模型文件中的污染物，默认是 O3
    """
    # 从原始的nc文件中获取投影信息
    nc_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/HR2DAY_LST_ACONC_v532_cb6r3_ae7_aq_WR413_MYR_STAGE_2011_12US1_2011.nc"
    ds_model = pyrsig.open_ioapi(nc_file)
    proj = pyproj.Proj(ds_model.crs_proj4)

    # 读取模型文件（CSV格式）
    df_model = pd.read_csv(model_file)
    # 检查 df_model 是否包含 model 列
    if'model' not in df_model.columns:
        raise ValueError("The model file does not contain the'model' column.")
    # 对模型文件的ROW和COL列都减去0.5,转换为中心点
    df_model.loc[:, ['ROW', 'COL']] = df_model[['ROW', 'COL']] - 0.5

    # 读取监测文件
    df_obs = pd.read_csv(monitor_file)
    # 过滤掉包含空值的行,因为W126的AtF可能会因为<75%数据为空
    df_obs = df_obs.dropna(subset=["Site", "Lat", "Lon", "O3"])

    nn = nna_methods.NNA(method="voronoi", k=30)

    # 无需按日期分组
    df_obs_grouped = df_obs.groupby(["Site"]).aggregate({"O3": "mean", "Lat": "mean", "Lon": "mean"}).reset_index()

    # 读取包含 Is 列的数据表
    region_df = pd.read_csv(region_table_file)

    # 筛选出 Is 列值为 1 的行
    us_region_df = region_df[region_df['Is'] == 1].copy()
    # -0.5确保后面的计算是正确的
    us_region_df.loc[:, ['COL', 'ROW']] = us_region_df[['COL', 'ROW']] - 0.5
    # 输入为x,y形式，对于model的col和row
    us_region_row_col = us_region_df[['COL', 'ROW']].values

    df_obs_grouped["x"], df_obs_grouped["y"] = proj(df_obs_grouped["Lon"], df_obs_grouped["Lat"])

    # 将 df_model 转换为 xarray
    df_model_xr = df_model.set_index(['ROW', 'COL'])[model_pollutant].to_xarray()

    # 用新的方式赋值 mod
    df_obs_grouped["mod"] = df_model_xr.sel(
        ROW=df_obs_grouped["y"].to_xarray(),
        COL=df_obs_grouped["x"].to_xarray(),
        method="nearest"
    )

    df_obs_grouped["bias"] = df_obs_grouped["mod"] - df_obs_grouped["O3"]
    df_obs_grouped["r_n"] = df_obs_grouped["O3"] / df_obs_grouped["mod"]

    df_prediction = df_model[["ROW", "COL"]].copy()
    # 通过训练使得网格点能快速搜寻临近点
    nn.fit(
        df_obs_grouped[["x", "y"]],
        df_obs_grouped[[monitor_pollutant, "mod", "bias", "r_n"]]
    )

    # 并行计算部分
    njobs = multiprocessing.cpu_count()
    zdf = nn.predict(us_region_row_col, njobs=njobs)

    # 创建一个全为 NaN 的 DataFrame 用于存储预测结果
    result_df = pd.DataFrame(np.nan, index=df_prediction.index, columns=["vna_ozone", "vna_mod", "vna_bias", "vna_r_n"])

    # 将美国区域的预测结果填充到对应的行
    result_df.loc[us_region_df.index] = zdf

    df_prediction = pd.concat([df_prediction, result_df], axis=1)

    df_fusion = df_prediction.set_index(["ROW", "COL"]).to_xarray()

    # 获取 ROW 和 COL 的唯一值
    unique_rows = df_model['ROW'].unique()
    unique_cols = df_model['COL'].unique()

    # 重塑数据
    model_data = df_model.set_index(["ROW", "COL"])["model"].unstack()

    # 确保数据形状和维度匹配
    if model_data.shape == (len(unique_rows), len(unique_cols)):
        df_fusion["model"] = (("ROW", "COL"), model_data.values)
    else:
        raise ValueError("The shape of the model data does not match the dimensions.")

    df_fusion["avna_ozone"] = df_fusion["model"] - df_fusion["vna_bias"]

    # 这里对 reshaped_vna_r_n 进行正确的二维重塑
    reshaped_vna_r_n = df_prediction["vna_r_n"].values.reshape(len(unique_rows), len(unique_cols))
    # 显式提取数据
    evna_ozone_data = (df_fusion["model"] * reshaped_vna_r_n).data
    df_fusion["evna_ozone"] = (("ROW", "COL"), evna_ozone_data)

    df_fusion = df_fusion.to_dataframe().reset_index()
    df_fusion["model"] = df_model['model'].values.flatten()
    df_fusion["COL"] = (df_fusion["COL"] + 0.5).astype(int)
    df_fusion["ROW"] = (df_fusion["ROW"] + 0.5).astype(int)
    
    # 添加Period列，值为W126
    df_fusion["Period"] = "W126"

    df_fusion.to_csv(file_path, index=False)
    project_name = os.path.basename(file_path).replace(".csv", "")
    print(f"Data Fusion is done, the results are saved to {file_path}")
    return df_fusion


def process_multiple_years(years, base_save_path, base_model_path, base_monitor_path, region_table_file, 
                          monitor_pollutant="O3", model_pollutant="model"):
    """
    处理多年数据的主函数
    
    参数:
    years: 年份列表，如 [2002, 2003, 2004, 2007]
    base_save_path: 保存结果的基础路径
    base_model_path: 模型文件的基础路径
    base_monitor_path: 监测文件的基础路径
    region_table_file: 区域表文件路径
    monitor_pollutant: 监测数据中的污染物名称
    model_pollutant: 模型数据中的污染物名称
    """
    # 确保保存路径存在
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)
    
    # 对每年数据进行处理
    for year in years:
        print(f"\nProcessing year {year}...")
        
        # 构建各文件路径
        model_file = os.path.join(base_model_path, f"{year}_W126_EQUATES.csv")
        monitor_file = os.path.join(base_monitor_path, f"{year}_W126_Monitor.csv")
        daily_output_path = os.path.join(base_save_path, f"{year}_W126_AtF.csv")
        
        # 检查输入文件是否存在
        if not os.path.exists(model_file):
            print(f"Warning: Model file {model_file} not found, skipping year {year}")
            continue
            
        if not os.path.exists(monitor_file):
            print(f"Warning: Monitor file {monitor_file} not found, skipping year {year}")
            continue
        
        # 处理当前年份数据
        start_daily_data_fusion(
            model_file,
            monitor_file,
            region_table_file,
            daily_output_path,
            monitor_pollutant=monitor_pollutant,
            model_pollutant=model_pollutant
        )
    
    print("\nAll years processing completed!")


# 在 main 函数中调用
if __name__ == "__main__":
    # 配置参数
    years = [2019]  # 要处理的年份列表
    base_save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF"
    base_model_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF"
    base_monitor_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF"
    region_table_file = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/Region_CONUSHarvard.csv"
    
    # 处理多年数据
    process_multiple_years(
        years,
        base_save_path,
        base_model_path,
        base_monitor_path,
        region_table_file
    )