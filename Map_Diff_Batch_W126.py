#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import cmaps

# for show maps
from esil.rsm_helper.model_property import model_attribute
from esil.map_helper import get_multiple_data, show_maps

cmap_conc = cmaps.WhiteBlueGreenYellowRed
cmap_delta = cmaps.ViBlGrWhYeOrRe


def extract_key_period(period):
    """
    Extract key period (e.g., JFM, AMJ) from the full period string.
    """
    key_periods = ['W126']
    for key in key_periods:
        if key in period:
            return key
    return None


def get_year(filename):
    """
    Extract the year from the filename (assuming the year is in the range 2011 - 2020).
    """
    match = re.search(r"(20[0-2][0-9])", filename)
    if match:
        return match.group(0)
    return None


def get_dataset_label(variable, filename):
    """
    Get the dataset label based on variable and filename.
    """
    if variable == 'vna_ozone':
        return "VNA"
    elif variable == 'evna_ozone':
        return "eVNA"
    elif variable == 'avna_ozone':
        return "aVNA"
    elif variable == 'ds_ozone':
        return "Downscaler"
    elif variable =='model':
        return "EQUATES"
    elif variable == 'harvard_ml':
        return "Harvard ML"
    return "unknown"


def crop_image(image_path, crop_box):
    """
    裁剪图片
    :param image_path: 图片路径
    :param crop_box: 裁剪框，格式为 (left, upper, right, lower)
    :return: 裁剪后的图片
    """
    image = Image.open(image_path)
    cropped_image = image.crop(crop_box)
    return cropped_image


def merge_images(image_paths, output_path, crop_box=None, spacing=10):
    """
    横向合并多张图片为一张图片，并设置图片之间的间距，用白色填充。

    :param image_paths: 单张图片的路径列表
    :param output_path: 合并后图片的保存路径
    :param crop_box: 裁剪框，格式为 (left, upper, right, lower)，默认为不裁剪
    :param spacing: 图片之间的间距，单位为像素，默认为 10
    """
    # 读取并裁剪所有图片
    images = []
    for path in image_paths:
        if crop_box:
            image = crop_image(path, crop_box)
        else:
            image = Image.open(path)
        images.append(image)

    # 获取图片的宽度和高度
    width, height = images[0].size

    # 计算合并后图片的总宽度，考虑间距
    total_width = width * len(images) + spacing * (len(images) - 1)

    # 创建合并图片，宽度为所有图片宽度与间距之和，高度为单张图片的高度
    merged_image = Image.new('RGB', (total_width, height), color='white')

    # 粘贴每张图片到合并图片的对应位置，并添加间距
    current_x = 0
    for i, image in enumerate(images):
        merged_image.paste(image, (current_x, 0))
        current_x += width + spacing

    # 保存合并图片
    merged_image.save(output_path)
    print(f"合并图片已保存到 {output_path}")


def plot_us_map(
        fusion_output_files_x,
        fusion_output_files_y,
        model_file,
        save_path=None,
        boundary_json_file="/DeepLearning/mnt/Devin/boundary/USA_State.json",
        comparisons=None,
        variable_settings=None,
        selected_periods=None
):
    """
    description: plot the US map with different data fusion results
    @param {list} fusion_output_files_x: the list of data fusion output files for the x - axis
    @param {list} fusion_output_files_y: the list of data fusion output files for the y - axis
    @param {string} model_file: the model file used for data fusion
    @param {string} save_path: the path to save the plot
    @param {string} boundary_json_file: the boundary file for the US map
    @param {list} comparisons: list of tuples, each tuple contains two variable names for comparison
    @param {dict} variable_settings: dictionary of settings for each comparison
    @param {list} selected_periods: 要画图的指标列表，如 ["DJF", "JJA"]
    @return None
    """
    # Load model attributes
    mp = model_attribute(model_file)
    proj, longitudes, latitudes = mp.projection, mp.lons, mp.lats

    for fusion_output_file_x, fusion_output_file_y in zip(fusion_output_files_x, fusion_output_files_y):
        # Read both x - axis and y - axis data
        df_data_x = pd.read_csv(fusion_output_file_x)
        df_data_y = pd.read_csv(fusion_output_file_y)

        print("Columns in df_data_x:", df_data_x.columns)
        print("Columns in df_data_y:", df_data_y.columns)

        # Check if the required columns exist
        if "Period" not in df_data_x.columns or "Period" not in df_data_y.columns:
            print("The data fusion files do not contain the Period column!")
            continue

        # Extract key periods (e.g., JFM, AMJ) from the Period column
        df_data_x["Key_Period"] = df_data_x["Period"].apply(extract_key_period)
        df_data_y["Key_Period"] = df_data_y["Period"].apply(extract_key_period)

        # Drop rows where Key_Period is None
        df_data_x = df_data_x.dropna(subset=["Key_Period"])
        df_data_y = df_data_y.dropna(subset=["Key_Period"])

        # Debug: Print unique Key_Periods
        print("X - axis Key_Periods:", df_data_x["Key_Period"].unique())
        print("Y - axis Key_Periods:", df_data_y["Key_Period"].unique())

        # Extract year from filenames
        year_x = get_year(fusion_output_file_x)
        year_y = get_year(fusion_output_file_y)

        if year_x != year_y:
            print("Warning: The years in the input files do not match!")
            continue

        year = year_x

        # Create save path
        # save_path = os.path.join("/DeepLearning/mnt/shixiansheng/data_fusion/output/CompareMap_W126", f"{year}_CompareMap_W126")
        save_path = os.path.join("/DeepLearning/mnt/shixiansheng/data_fusion/output/CompareMap", f"{year}_CompareMap")
        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                print(f"Created directory: {save_path}")
            else:
                print(f"Directory already exists: {save_path}")
        except OSError as e:
            print(f"Error creating directory: {e}")
            continue

        # 创建 Merged_Diff 文件夹
        save_path1 = "/DeepLearning/mnt/shixiansheng/data_fusion/output"
        merged_diff_path = os.path.join(save_path1, "Merged_HourlyIntoMetrics_Methoddiff_W126AtF")
        try:
            if not os.path.exists(merged_diff_path):
                os.makedirs(merged_diff_path)
                print(f"Created directory: {merged_diff_path}")
            else:
                print(f"Directory already exists: {merged_diff_path}")
        except OSError as e:
            print(f"Error creating Merged_Diff directory: {e}")
            continue

        print("Save path:", save_path)
        print("Merged_Diff path:", merged_diff_path)

        # 如果指定了画图指标，则只处理这些指标，否则处理所有指标
        if selected_periods:
            key_periods = [period for period in df_data_x["Key_Period"].unique() if period in selected_periods]
            print(f"Selected key periods: {key_periods}")
        else:
            key_periods = df_data_x["Key_Period"].unique()

        for key_period in tqdm(key_periods):
            print(f"Processing key period: {key_period}")
            # Filter data for the current key period
            df_x_period = df_data_x[df_data_x["Key_Period"] == key_period]
            df_y_period = df_data_y[df_data_y["Key_Period"] == key_period]

            print(f"Number of rows in df_x_period for {key_period}:", len(df_x_period))
            print(f"Number of rows in df_y_period for {key_period}:", len(df_y_period))

            if len(df_x_period) == 0 or len(df_y_period) == 0:
                print(f"No data found for period {key_period} in one of the files!")
                continue

            # 按照规则分组比较,方式为右边减左边
            #有harvard_ml的group
            # groups = [
            #     [('model','vna_ozone'), ('model','evna_ozone'), ('model','avna_ozone'), ('model','ds_ozone'), ('model','harvard_ml')],
            #     [('vna_ozone', 'evna_ozone'), ('vna_ozone', 'avna_ozone'), ('vna_ozone', 'ds_ozone'), ('vna_ozone', 'harvard_ml'), ('ds_ozone', 'harvard_ml')],
            #     [('evna_ozone', 'avna_ozone'), ('evna_ozone', 'ds_ozone'), ('evna_ozone', 'harvard_ml'), ('avna_ozone', 'ds_ozone'), ('avna_ozone', 'harvard_ml')]
            # ]
            groups = [[('model','vna_ozone'), ('model','evna_ozone'), ('model','avna_ozone')],
                [('vna_ozone', 'evna_ozone'), ('vna_ozone', 'avna_ozone')],
                [('evna_ozone', 'avna_ozone')]
            ]


            for group_index, group in enumerate(groups):
                group_image_paths = []
                for i, (variable_x, variable_y) in enumerate(group):
                    if variable_x not in df_x_period.columns or variable_y not in df_y_period.columns:
                        print(f"Variable {variable_x} or {variable_y} not found in data files. Skipping...")
                        continue

                    # Get dataset labels
                    dataset_label_x = get_dataset_label(variable_x, fusion_output_file_x)
                    dataset_label_y = get_dataset_label(variable_y, fusion_output_file_y)

                    # Generate title
                    title = f"{year}_{key_period}: {dataset_label_y} - {dataset_label_x}"

                    # Reshape data to match the grid
                    grid_variable_x = df_x_period[variable_x].values.reshape(longitudes.shape)
                    grid_variable_y = df_y_period[variable_y].values.reshape(longitudes.shape)

                    # Calculate differences (y - x)
                    grid_variable_diff = grid_variable_y - grid_variable_x

                    # Get settings for the current comparison
                    comparison_key = f"{variable_y}-{variable_x}"
                    settings = variable_settings.get(comparison_key, variable_settings["default"])

                    # Prepare data for plotting
                    dict_data = {}
                    get_multiple_data(
                        dict_data,
                        dataset_name=title,
                        variable_name=f"",
                        grid_x=longitudes,
                        grid_y=latitudes,
                        grid_concentration=grid_variable_diff,
                        is_delta=settings.get('is_delta', True),
                        cmap=settings.get('cmap_delta', cmap_delta)
                    )

                    # Plot the maps
                    try:
                        fig = show_maps(
                            dict_data,
                            unit=settings.get('unit', "ppm-hrs"),
                            cmap=settings.get('cmap_conc', cmap_conc),
                            show_lonlat=settings.get('show_lonlat', True),
                            projection=proj,
                            is_wrf_out_data=settings.get('is_wrf_out_data', True),
                            boundary_file=boundary_json_file,
                            show_original_grid=settings.get('show_original_grid', True),
                            panel_layout=settings.get('panel_layout', None),
                            delta_map_settings=settings.get('delta_map_settings', {
                                "cmap": cmap_delta,
                                "value_range": (None, None),
                                "colorbar_ticks_num": None,
                                "colorbar_ticks_value_format": ".2f",
                                "value_format": ".2f"
                            }),
                            title_fontsize=settings.get('title_fontsize', 11),
                            xy_title_fontsize=settings.get('xy_title_fontsize', 9),
                            show_dependenct_colorbar=settings.get('show_dependenct_colorbar', True),
                            show_domain_mean=settings.get('show_domain_mean', True)
                        )
                    except Exception as e:
                        print(f"Error in show_maps for {title}: {e}")

                    # Save the plot with the new title
                    if save_path is not None:
                        save_file = os.path.join(save_path, f"{title}.png")
                        group_image_paths.append(save_file)
                        try:
                            fig.savefig(save_file, dpi=300, bbox_inches="tight")
                            print(f"The data fusion plot for {title} is saved to {save_file}")
                        except Exception as e:
                            print(f"Error saving single plot {save_file}: {e}")
                    plt.close(fig)

                # 合并该组的图片
                if group_image_paths:
                    group_title = '_'.join([f"{y}-{x}" for x, y in group])
                    group_merged_output_path = os.path.join(merged_diff_path, f"{year}_{key_period}_{group_title}.png")
                    # 假设裁剪框为 (0, 0, 800, 600)，你可以根据实际需求修改,600为纵向尺寸
                    merge_images(group_image_paths, group_merged_output_path, crop_box=(0, 0, 1345, 1030), spacing=200)

if __name__ == "__main__":
    # 输入文件列表

    #输入的每年FtA的W126文件
    # fusion_output_files_y = [
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2002_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2003_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2004_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2005_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2006_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2007_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2008_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2009_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2010_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2012_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2013_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2014_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2015_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2016_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2017_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2018_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2019_W126_ST_Limit.csv",
    #     # 添加更多年份的文件路径
    # ]
    # fusion_output_files_x = [
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2002_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2003_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2004_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2005_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2006_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2007_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2008_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2009_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2010_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2012_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2013_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2014_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2015_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2016_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2017_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2018_W126_ST_Limit.csv",
    #     r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2019_W126_ST_Limit.csv",
    #     # 添加更多年份的文件路径
    # ]

    #输入的AtF文件
    fusion_output_files_y = ['/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2002_W126_AtF.csv',
                    "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2003_W126_AtF.csv",
                    "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2004_W126_AtF.csv",
                    "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2005_W126_AtF.csv",
                    "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2006_W126_AtF.csv",
                    "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2007_W126_AtF.csv",
                    "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2008_W126_AtF.csv",
                "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2009_W126_AtF.csv",
                 "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2010_W126_AtF.csv",
                 '/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2011_W126_AtF.csv',
                 "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2012_W126_AtF.csv",
                  "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2013_W126_AtF.csv",
                  "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2014_W126_AtF.csv",
                   "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2015_W126_AtF.csv",
                   "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2016_W126_AtF.csv",
                   "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2017_W126_AtF.csv",
                   "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2018_W126_AtF.csv",
                   "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2019_W126_AtF.csv",]
    fusion_output_files_x = ['/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2002_W126_AtF.csv',
                    "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2003_W126_AtF.csv",
                    "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2004_W126_AtF.csv",
                    "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2005_W126_AtF.csv",
                    "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2006_W126_AtF.csv",
                    "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2007_W126_AtF.csv",
                    "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2008_W126_AtF.csv",
                "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2009_W126_AtF.csv",
                 "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2010_W126_AtF.csv",
                 '/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2011_W126_AtF.csv',
                 "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2012_W126_AtF.csv",
                  "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2013_W126_AtF.csv",
                  "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2014_W126_AtF.csv",
                   "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2015_W126_AtF.csv",
                   "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2016_W126_AtF.csv",
                   "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2017_W126_AtF.csv",
                   "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2018_W126_AtF.csv",
                   "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2019_W126_AtF.csv",]
    # 其他代码
    model_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/HR2DAY_LST_ACONC_v532_cb6r3_ae7_aq_WR413_MYR_STAGE_2011_12US1_2011.nc"

    variables = ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone']
    comparisons = list(itertools.combinations(variables, 2))

    # 为每个对比设置独立的绘图参数（统一设置）
    variable_settings = {
        'default': {
            'is_delta': True,
            'cmap_delta': cmaps.ViBlGrWhYeOrRe,
            'unit': "ppm-hrs",
            'cmap_conc': cmaps.WhiteBlueGreenYellowRed,
           'show_lonlat': True,
            'is_wrf_out_data': True,
           'show_original_grid': True,
            'panel_layout': None,
            'delta_map_settings': {
                "cmap": cmaps.ViBlGrWhYeOrRe,
                "value_range": (-10, 10),
                "colorbar_ticks_num": None,
                "colorbar_ticks_value_format": ".2f",
                "value_format": ".2f"
            },
            'title_fontsize': 11,
            'xy_title_fontsize': 9,
           'show_dependenct_colorbar': True,
           'show_domain_mean': True
        }
    }

    # 指定画图的指标
    selected_periods = ['W126']
    plot_us_map(
        fusion_output_files_x,
        fusion_output_files_y,
        model_file,
        comparisons=comparisons,
        variable_settings=variable_settings,
        selected_periods=selected_periods
    )
    print("Done!")


# In[ ]:




