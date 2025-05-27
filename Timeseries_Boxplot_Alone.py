import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as font_manager

# 设置字体为罗马字体（Serif）
plt.rcParams["font.family"] = ["serif", "Times New Roman", "DejaVu Serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['text.usetex'] = False  # 禁用LaTeX渲染，使用matplotlib内置渲染

def inspect_data_file(file_path):
    """检查单个数据文件的内容、结构和统计信息"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise Exception(f"读取文件失败: {e}")


def compare_merge_columns(df1, df2, merge_columns):
    """比较两个DataFrame的合并列，返回兼容性报告"""
    report = {}

    for col in merge_columns:
        if col in df1.columns and col in df2.columns:
            report[col] = {
                'dtype1': df1[col].dtype,
                'dtype2': df2[col].dtype,
                'unique1_count': df1[col].nunique(),
                'unique2_count': df2[col].nunique(),
                'common_values_count': len(set(df1[col].dropna().unique()) & set(df2[col].dropna().unique()))
            }
        else:
            report[col] = {'error': f'列 {col} 不存在于一个或两个数据集中'}

    return report


def check_model_column(df1, df2, col_name='model'):
    """检查model列在两个数据集中的值分布情况"""
    if col_name not in df1.columns or col_name not in df2.columns:
        return {'error': f'列 {col_name} 不存在'}

    return {
       'stats1': df1[col_name].describe().to_dict(),
       'stats2': df2[col_name].describe().to_dict(),
       'missing1': df1[col_name].isnull().sum(),
       'missing2': df2[col_name].isnull().sum(),
        'range1': (df1[col_name].min(), df1[col_name].max()),
        'range2': (df2[col_name].min(), df2[col_name].max()),
        'common_unique_count': len(set(df1[col_name].dropna().unique()) & set(df2[col_name].dropna().unique()))
    }


def merge_and_process_data(data_2002, data_delta, selected_periods=None, exclude_w126=True):
    """合并并处理数据，可选择特定Periods"""
    if data_2002 is None or data_delta is None:
        return None

    # 执行合并 - 使用outer连接，保留所有数据
    merge_columns = ['Period', 'ROW', 'COL']  # 不使用model列进行合并
    merged_data = pd.merge(
        data_2002,
        data_delta,
        on=merge_columns,
        suffixes=['_2002', '_delta'],
        how='outer'
    )

    # 排除W126相关数据（如果需要）
    if exclude_w126 and 'W126' in merged_data['Period'].unique():
        merged_data = merged_data[merged_data['Period'] != 'W126']

    # 筛选特定Periods
    if selected_periods:
        merged_data = merged_data[merged_data['Period'].isin(selected_periods)]

    return merged_data


def create_ozone_bins(data, ozone_columns):
    """根据2002年臭氧浓度创建分箱"""
    bins = [0, 40, 50, 60, 70, 80, float('inf')]
    labels = ['< 40', '40-50', '50-60', '60-70', '70-80', '> 80']

    for col in ozone_columns:
        col_2002 = f"{col}_2002"
        bin_col = f"{col}_bin"
        if col_2002 in data.columns:
            data[bin_col] = pd.cut(data[col_2002], bins=bins, labels=labels, include_lowest=True)

    return data


def reshape_data_for_boxplot(data, period, ozone_columns):
    """重塑数据，使每种方法的数据作为一个独立的类别"""
    period_data = data[data['Period'] == period].copy()
    if period_data.empty:
        return None

    reshaped_data = pd.DataFrame(columns=['bin', 'delta', 'method'])

    for method in ozone_columns:
        method_2002 = f"{method}_2002"
        method_delta = f"{method}_delta"
        method_bin = f"{method}_bin"

        if method_2002 in period_data.columns and method_delta in period_data.columns:
            method_data = period_data[[method_bin, method_delta]].copy()
            method_data.columns = ['bin', 'delta']
            method_data['method'] = method
            reshaped_data = pd.concat([reshaped_data, method_data], ignore_index=True)

    # 去除无效数据
    reshaped_data = reshaped_data.dropna(subset=['delta', 'bin'])
    if reshaped_data.empty:
        return None

    return reshaped_data


def plot_boxplots(data, output_dir='/DeepLearning/mnt/shixiansheng/data_fusion/output/boxplots'):
    """为每个Period绘制箱线图，每个臭氧区间包含多个方法的箱子"""
    if data is None or data.empty:
        return

    os.makedirs(output_dir, exist_ok=True)
    periods = data['Period'].unique()

    # 定义方法顺序和颜色映射
    ozone_columns = ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone', 'ds_ozone']
    method_colors = {
       'model': '#d62728',      # 红色
        'vna_ozone': '#1f77b4',  # 蓝色
        'evna_ozone': '#ff7f0e',  # 橙色
        'avna_ozone': '#2ca02c',  # 绿色
        'ds_ozone': '#9467bd',    # 紫色
    }
    # 使用Unicode下标字符₃
    method_labels = {
       'model': 'EQUATES',
        'vna_ozone': 'VNA',
        'evna_ozone': 'eVNA',
        'avna_ozone': 'aVNA',
        'ds_ozone': 'Downscaler'
    }

    for period in periods:
        reshaped_data = reshape_data_for_boxplot(data, period, ozone_columns)
        if reshaped_data is None:
            continue

        plt.figure(figsize=(14, 7))  # 调整图像宽度以适应右侧图例
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['legend.fontsize'] = 14

        # 绘制箱线图，设置离群点样式
        ax = sns.boxplot(
            x='bin',
            y='delta',
            hue='method',
            width = 0.5,
            data=reshaped_data,
            palette=method_colors,
            showfliers=True,  # 显示离群点
            flierprops={
                'marker': 'o',          # 圆点标记
                'markerfacecolor': 'white',  # 白色填充
                'markeredgecolor': 'black',  # 黑色边缘
                'markersize': 5,        # 标记大小
                'alpha': 0.6            # 透明度
            },
            order=['< 40', '40-50', '50-60', '60-70', '70-80', '> 80']
        )

        # 设置标题和标签（使用Unicode下标字符₃）
        plt.title(f'{period}: 2019-2002 Ozone Change', pad=20)
        plt.xlabel('2002 O₃ (ppbv)', labelpad=15)
        plt.ylabel('2019-2002 O₃ (ppbv)', labelpad=15)

        # 设置Y轴范围
        # plt.ylim(-140, 30) #top-10
        plt.ylim(-40, 22.8) #DJF etc

        # 添加水平线表示y=0
        # plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)

        # 调整图例位置到图的右侧
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            plt.legend(
                handles=handles,
                labels=[method_labels.get(l, l) for l in labels],
                title="Method",
                loc='center left',  # 将图例放在中心左侧（实际会在右侧）
                bbox_to_anchor=(1, 0.5),  # 调整图例位置
                frameon=True,
                framealpha=0.9,
                edgecolor='None',
                title_fontsize=14,
                labelspacing=0.5
            )

        # 优化图形外观
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()  # 自动调整布局
        plt.savefig(os.path.join(output_dir, f"ozone_change_{period}.png"), dpi=300, bbox_inches='tight')
        plt.close()


def plot_boxplots_alone(data, output_dir='/DeepLearning/mnt/shixiansheng/data_fusion/output/boxplots_Alone'):
    """为每个Period和每个方法单独绘制箱线图"""
    if data is None or data.empty:
        return

    os.makedirs(output_dir, exist_ok=True)
    periods = data['Period'].unique()

    # 定义方法顺序和颜色映射
    ozone_columns = ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone', 'ds_ozone']
    method_colors = {
       'model': '#d62728',      # 红色
        'vna_ozone': '#1f77b4',  # 蓝色
        'evna_ozone': '#ff7f0e',  # 橙色
        'avna_ozone': '#2ca02c',  # 绿色
        'ds_ozone': '#9467bd',    # 紫色
    }
    # 使用Unicode下标字符₃
    method_labels = {
       'model': 'EQUATES',
        'vna_ozone': 'VNA',
        'evna_ozone': 'eVNA',
        'avna_ozone': 'aVNA',
        'ds_ozone': 'Downscaler'
    }

    for period in periods:
        period_data = data[data['Period'] == period].copy()
        if period_data.empty:
            continue

        for method in ozone_columns:
            method_2002 = f"{method}_2002"
            method_delta = f"{method}_delta"
            method_bin = f"{method}_bin"

            if method_2002 not in period_data.columns or method_delta not in period_data.columns:
                continue

            # 提取该方法的数据
            method_data = period_data[[method_bin, method_delta]].copy()
            method_data.columns = ['bin', 'delta']
            
            # 去除无效数据
            method_data = method_data.dropna(subset=['delta', 'bin'])
            if method_data.empty:
                continue

            plt.figure(figsize=(12, 7))
            plt.rcParams['font.size'] = 14
            plt.rcParams['axes.labelsize'] = 16
            plt.rcParams['axes.titlesize'] = 18

            # 绘制箱线图
            ax = sns.boxplot(
                x='bin',
                y='delta',
                data=method_data,
                color=method_colors[method],
                showfliers=True,
                flierprops={
                    'marker': 'o',
                    'markerfacecolor': 'white',
                    'markeredgecolor': 'black',
                    'markersize': 5,
                    'alpha': 0.6
                },
                order=['< 40', '40-50', '50-60', '60-70', '70-80', '> 80']
            )

            # 设置标题和标签
            plt.title(f'{period}: {method_labels[method]} - 2019-2002 Ozone Change', pad=20)
            plt.xlabel('2002 O₃ (ppbv)', labelpad=15)
            plt.ylabel('2019-2002 O₃ (ppbv)', labelpad=15)

            # 设置Y轴范围
            plt.ylim(-140, 30)
            plt.ylim(-40, 22.8)  # DJF etc

            # 添加水平线表示y=0
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)

            # 优化图形外观
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"ozone_change_{period}_{method}.png"), dpi=300, bbox_inches='tight')
            plt.close()


def main():
    """主函数"""
    try:
        # 文件路径
        data_2002_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/DailyData_WithoutCV/2002_Data_WithoutCV_Metrics.csv"
        data_delta_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/DailyData_WithoutCV_Delta/2019-2002_Data_WithoutCV_Metrics.csv"

        # 读取数据
        df_2002 = inspect_data_file(data_2002_path)
        df_delta = inspect_data_file(data_delta_path)

        # 比较合并列
        merge_columns = ['Period', 'ROW', 'COL', 'model']
        merge_report = compare_merge_columns(df_2002, df_delta, merge_columns)

        # 检查model列
        model_report = check_model_column(df_2002, df_delta)

        # 指定要绘制的Periods
        period_columns = ['DJF','MAM', 'JJA', 'SON', 'Annual','Apr-Sep']
        # period_columns = ['top-10']

        # 合并和处理数据
        merged_data = merge_and_process_data(df_2002, df_delta, selected_periods=period_columns)

        if merged_data is None or merged_data.empty:
            raise Exception("处理后没有数据可用于分析")

        # 创建臭氧分箱
        ozone_columns = ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone', 'ds_ozone']
        binned_data = create_ozone_bins(merged_data, ozone_columns)

        # 绘制原始的合并箱线图
        plot_boxplots(binned_data)
        
        # 绘制单独的箱线图
        plot_boxplots_alone(binned_data)

        print(f"已完成绘制指定Periods: {period_columns} 的箱线图")

    except Exception as e:
        print(f"程序执行出错: {e}")


if __name__ == "__main__":
    main()
