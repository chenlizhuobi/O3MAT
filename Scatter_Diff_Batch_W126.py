import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, linregress
import itertools
import re
from PIL import Image

# -------------------- 工具函数 --------------------
def extract_key_period(period):
    """
    Extract key period (e.g., JFM, AMJ) from the full period string.
    """
    key_periods = ["DJF", "MAM", "JJA", "SON", 'Annual', 'Apr-Sep', 'W126', 'W126']
    for key in key_periods:
        if key in period:
            return key
    return None


def get_year(filename):
    """
    从文件名中提取年份（假设年份在 2000 - 2030 范围内）。
    """
    match = re.search(r"(20[0-2][0-9])", filename)
    if match:
        return match.group(1)
    return None


def get_axis_label(filename, variable):
    """
    根据文件名和变量生成轴标签。
    - 如果变量为 'model'，根据文件名判断是 'Harvard ML' 还是 'EQUATES'。
    - 其他情况根据变量名确定标签，如 'vna_ozone' 对应 'VNA'。
    """
    if 'model' in variable:
        return "EQUATES"
    elif "harvard_ml" in variable:
        return "Harvard ML"
    elif "evna_ozone" in variable:
        return "eVNA"
    elif "avna_ozone" in variable:
        return "aVNA"
    elif "ds_ozone" in variable:
        return "Downscaler"
    elif "vna_ozone" in variable:
        return "VNA"
    elif "Conc" in variable:
        return "Monitor"
    elif "O3" in variable:
        return "Monitor (Hourly)"
    return "unknown"


# -------------------- 图片合并函数 --------------------
def merge_images(image_paths, output_path, spacing=20):
    """
    横向合并多张图片为一张图片，并设置图片之间的间距，用白色填充。

    :param image_paths: 单张图片的路径列表
    :param output_path: 合并后图片的保存路径
    :param spacing: 图片之间的间距，单位为像素，默认为 20
    """
    # 读取所有图片
    images = [Image.open(path) for path in image_paths]

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


# -------------------- 定义绘图函数 --------------------
def plot_density_scatter(dataframe_x, dataframe_y, x_column, y_column, period_column, output_dir_base, year, period_value, x_axis_filename, y_axis_filename, output_csv_path, default_range=None, W126_range=None, W126_special_years=None):
    """
    绘制散点密度图：x_column vs y_column。
    文件名包含对应的 Period 字段。
    """
    # 获取数据（通过关键字匹配 Period）
    df_period_x = dataframe_x[dataframe_x[period_column].str.contains(period_value, case=False, na=False)]
    df_period_y = dataframe_y[dataframe_y[period_column].str.contains(period_value, case=False, na=False)]

    # 如果数据为空，跳过
    if df_period_x.empty or df_period_y.empty:
        print(f"数据中没有有效数据，跳过 Period: {period_value} 的绘图。")
        return

    # 获取数据
    x_data = df_period_x[x_column].values
    y_data = df_period_y[y_column].values

    # 打印正在处理的数据信息
    print(f"正在处理文件: {y_axis_filename}, Period: {period_value}")
    print(f"x_data 长度: {len(x_data)}, y_data 长度: {len(y_data)}")
    print(f"x_data 前5个值: {x_data[:5]}")
    print(f"y_data 前5个值: {y_data[:5]}")
    print(f"x_data 中 NaN 的数量: {np.isnan(x_data).sum()}")
    print(f"y_data 中 NaN 的数量: {np.isnan(y_data).sum()}")

    # 核密度计算
    valid_indices = ~np.isnan(x_data) & ~np.isnan(y_data)
    valid_x_data = x_data[valid_indices]
    valid_y_data = y_data[valid_indices]

    if len(valid_x_data) == 0 or len(valid_y_data) == 0:
        print(f"数据为空，跳过 Period: {period_value} 的绘图。")
        return

    # 只提取季节和年份部分
    period_search = extract_key_period(period_value)
    if period_search:
        period_value = period_search

    # 拼接成期望的格式，如：2011_JAS
    formatted_period = f"{year}_{period_value}"

    # 生成轴标签
    x_label = get_axis_label(x_axis_filename, x_column)  # x 轴标签
    y_label = get_axis_label(y_axis_filename, y_column)  # y 轴标签

    full_title = f"{formatted_period}: {y_label} vs. {x_label}"

    # -------------------- 修改：检查文件是否已存在 --------------------
    # 创建年份文件夹（如 2011_CompareScatter）
    year_folder = f"{year}_CompareScatter"
    output_dir = os.path.join(output_dir_base, year_folder)
    
    # 如果路径不存在，则自动创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建年份文件夹: {output_dir}")
    
    # 保存图像，文件名包含 Period 字段和输入文件名（不含路径）
    output_file_name = f'{full_title}.png'
    output_path = os.path.join(output_dir, output_file_name)
    
    # 检查文件是否已存在
    if os.path.exists(output_path):
        print(f"文件已存在，跳过生成: {output_path}")
        return output_path, formatted_period, (x_column, y_column)

    # 核密度计算
    xy = np.vstack([valid_x_data, valid_y_data])
    kde = gaussian_kde(xy)
    z = kde(xy)
    z = (z - z.min()) / (z.max() - z.min())  # 归一化

    # 绘制散点密度图
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(valid_x_data, valid_y_data, c=z, cmap='jet', s=20, alpha=0.8)
    fig.colorbar(scatter, ax=ax)  # 删除了 label 参数

    # -------------------- 坐标轴范围设置 --------------------
    if period_value == 'W126' and W126_range:
        # W126 周期的特殊处理
        if W126_special_years and year in W126_special_years:
            # 特殊年份的特殊范围
            ax.set_xlim(W126_special_years[year])
            ax.set_ylim(W126_special_years[year])
            print(f"使用 W126 特殊年份设置的坐标轴范围: {W126_special_years[year]} for {year}")
        else:
            # W126 周期的默认范围
            ax.set_xlim(W126_range)
            ax.set_ylim(W126_range)
            print(f"使用 W126 默认坐标轴范围: {W126_range}")
    else:
        # 其他周期使用统一的默认范围
        if default_range:
            ax.set_xlim(default_range)
            ax.set_ylim(default_range)
            print(f"使用默认坐标轴范围: {default_range}")
        else:
            # 如果未提供默认范围，使用原逻辑
            max_val = max(valid_x_data.max(), valid_y_data.max())
            max_val = 65  # 原代码中的固定值
            max_val1 = max_val + 3
            ax.set_xlim(-3, max_val1)
            ax.set_ylim(-3, max_val1)
            print(f"使用原始默认坐标轴范围: (-3, {max_val1})")

    # 确保1:1线适应新的坐标轴范围
    x_min, x_max = ax.get_xlim()
    ax.plot([x_min, x_max], [x_min, x_max], 'b--', lw=0.5)  # 蓝色 1:1 线

    # 添加回归线
    slope, intercept, r_value, _, _ = linregress(valid_x_data, valid_y_data)
    r_squared = r_value ** 2
    mb = np.mean(valid_y_data - valid_x_data)  # 计算 MB (平均偏差)

    # 计算 RMSE
    rmse = np.sqrt(np.mean((valid_y_data - valid_x_data) ** 2))  # 计算 RMSE

    # 调整回归线方程格式
    if intercept >= 0:
        regression_equation = fr"$y = {slope:.2f}\it{{x}} + {intercept:.2f}$"
    else:
        regression_equation = fr"$y = {slope:.2f}\it{{x}} {intercept:.2f}$"

    # 将 R²、MB、RMSE 和拟合直线方程添加到左上角
    r_squared = round(r_squared, 2)
    mb = round(mb, 2)
    rmse = round(rmse, 2)

    # 绘制回归线（红色）
    regression_line = slope * np.array([x_min, x_max]) + intercept
    ax.plot([x_min, x_max], regression_line, 'r-', lw=0.5, label="Regression Line")
    slope = round(slope, 2)
    
    # 将统计信息添加到左上角
    ax.text(0.05, 0.95, f"$R^2$ = {r_squared:.2f}\nMB = {mb}\nRMSE = {rmse}\n{regression_equation}",
            transform=ax.transAxes, ha="left", va="top", fontsize=12, fontname='Times New Roman', color='black')

    # 设置标题和标签
    ax.set_xlabel(x_label, fontsize=12, fontname='Times New Roman')  # x 轴标签
    ax.set_ylabel(y_label, fontsize=12, fontname='Times New Roman')  # y 轴标签

    # 将标题放置到图像顶部
    fig.subplots_adjust(top=0.85)  # 调整标题的位置
    ax.set_title(full_title, fontsize=13, loc='center', fontname='Times New Roman')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)  # 关闭图形以释放内存
    print(f"散点密度图已保存至 {output_path}")

    # 返回保存的图片路径，用于后续合并
    return output_path, formatted_period, (x_column, y_column)


# -------------------- 按周期和组合并图片 --------------------
def merge_images_by_period_and_group(all_period_images, output_dir, groups, spacing=20):
    """
    按周期和指定的变量组合并图片
    
    :param all_period_images: 所有图片信息，按周期分组
    :param output_dir: 输出目录
    :param groups: 定义的变量组
    :param spacing: 图片间距
    """
    # 为每个组创建一个文件夹
    for group_idx, group in enumerate(groups):
        group_dir = os.path.join(output_dir, f"Group_{group_idx+1}")
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)
            print(f"创建组文件夹: {group_dir}")
        
        # 按周期处理
        for period, image_info_list in all_period_images.items():
            # 筛选属于当前组的图片
            group_images = []
            for info in image_info_list:
                path, year, (x_col, y_col) = info
                # 检查这对变量是否在当前组中
                if (x_col, y_col) in group or (y_col, x_col) in group:
                    group_images.append(info)
            
            # 如果找到了属于该组的图片，则合并它们
            if group_images:
                image_paths = [info[0] for info in group_images]
                # 生成组名
                group_name = "_vs_".join([f"{get_axis_label('', pair[0])}_{get_axis_label('', pair[1])}" for pair in group])
                output_path = os.path.join(group_dir, f"Merged_{period}_{group_name}.png")
                
                # -------------------- 修改：检查合并图片是否已存在 --------------------
                if os.path.exists(output_path):
                    print(f"合并图片已存在，跳过生成: {output_path}")
                    continue
                
                print(f"正在合并 {period} 的组 {group_idx+1} 图片: {image_paths}")
                merge_images(image_paths, output_path, spacing=spacing)


# -------------------- 读取和处理多个文件 --------------------
def process_file(fusion_output_file, x_axis_file, all_period_images=None, default_range=None, W126_range=None, W126_special_years=None):
    # 读取第一个文件的数据（y 轴数据）
    df_data_y = pd.read_csv(fusion_output_file)

    # 读取第二个文件的数据（x 轴数据）
    df_data_x = pd.read_csv(x_axis_file)

    # 提取年份
    year = get_year(fusion_output_file)
    if not year:
        print(f"无法从文件名 {fusion_output_file} 中提取年份，跳过处理")
        return

    # 提取Period列
    period_column = 'Period'  # Period列

    variables = ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone']
    comparisons = list(itertools.combinations(variables, 2))

    keywords = ['W126']
    # keywords = ['W126']

    # 动态生成基础输出路径
    output_dir_base = "/DeepLearning/mnt/shixiansheng/data_fusion/output/CompareScatter"

    # 如果基础路径不存在，则自动创建
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)
        print(f"创建基础输出目录: {output_dir_base}")

    output_csv_path = os.path.join(output_dir_base, 'Metrics.csv')

    # 用于存储该文件生成的所有图片路径，按周期分组
    file_period_images = {}

    # 遍历每个关键字并绘制图形
    for keyword in keywords:
        for x_column, y_column in comparisons:
            image_path = plot_density_scatter(
                df_data_x, df_data_y, x_column, y_column, period_column, 
                output_dir_base, year, keyword, x_axis_file, fusion_output_file, output_csv_path,
                default_range=default_range,
                W126_range=W126_range,
                W126_special_years=W126_special_years
            )

            if image_path:
                path, period, var_pair = image_path
                # 将图片路径添加到按周期分组的字典中
                if period not in file_period_images:
                    file_period_images[period] = []
                file_period_images[period].append((path, year, var_pair))

                # 如果提供了全局存储字典，则也添加到其中
                if all_period_images is not None:
                    if period not in all_period_images:
                        all_period_images[period] = []
                    all_period_images[period].append((path, year, var_pair))

    # 返回该文件生成的图片信息
    return file_period_images


if __name__ == "__main__":
    # 输入文件列表（多年数据）,AtF文件
    fusion_output_files = [
        '/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF/2002_W126_AtF.csv',
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

    # 对应的x轴文件列表（与fusion_output_files一一对应）
    x_axis_files = fusion_output_files  # 假设x轴文件与y轴文件相同


    # 输入文件列表（多年数据）,AtF文件
    
    # fusion_output_files = [
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

    # x_axis_files = fusion_output_files  # 假设x轴文件与y轴文件相同

    # 用于存储所有年份、所有周期的图片路径
    all_period_images = {}

    # -------------------- 定义坐标轴范围 --------------------
    # 所有周期的默认范围（除了W126）
    default_range = (-3, 90)  # 与原代码逻辑一致
    
    # W126 周期的默认范围
    W126_range = (-3, 90)
    
    # W126 周期中特殊年份的范围
    W126_special_years = {
        # '2002': (-3, 233),  # 2012年的特殊范围
        # '2003': (-3, 233),
        # '2004': (-3, 233),
        # '2005': (-3, 233),
        # '2006': (-3, 233),
        # '2007': (-3, 233),
        # '2009': (-3, 233),     # 2013年的特殊范围
    }

    # 处理每个文件
    for fusion_file, x_file in zip(fusion_output_files, x_axis_files):
        print(f"\n处理文件: {fusion_file}")
        process_file(
            fusion_file, x_file, all_period_images,
            default_range=default_range,
            W126_range=W126_range,
            W126_special_years=W126_special_years
        )

    # 创建合并图片的输出目录
    merged_output_dir = os.path.join('/DeepLearning/mnt/shixiansheng/data_fusion/output', "Merged_Scatter_W126_AtF")
    #确认后再改成原路径
    if not os.path.exists(merged_output_dir):
        os.makedirs(merged_output_dir)
        print(f"创建合并图片目录: {merged_output_dir}")

    #定义合并组

    groups = [[('model','vna_ozone'), ('model','evna_ozone'), ('model','avna_ozone')],
             [('vna_ozone', 'evna_ozone'), ('vna_ozone', 'avna_ozone')],
             [('evna_ozone','avna_ozone')]
    ]

    # 按周期和组合并图片
    merge_images_by_period_and_group(all_period_images, merged_output_dir, groups, spacing=30)

    print("所有图片处理和合并完成！")    