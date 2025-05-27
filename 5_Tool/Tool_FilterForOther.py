import pandas as pd


def read_csv_file(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误：未找到指定的 CSV 文件 {file_path}，请检查文件路径是否正确。")
        return None


def filter_data(df_is, df_input):
    try:
        # 从带 Is 的表中提取 Is = 1 的 ROW 和 COL
        filter_df = df_is[df_is['Is'] != 1][['ROW', 'COL']]
        # 使用 merge 函数进行条件筛选
        merged = pd.merge(df_input, filter_df, on=['ROW', 'COL'], how='left', indicator=True)
        columns_to_update = ['model','vna_ozone','evna_ozone','avna_ozone']
        df_input.loc[merged['_merge'] == 'both', columns_to_update] = None
        return df_input
    except KeyError:
        print("错误：数据文件中缺少必要的列，请检查列名是否正确。")
        return None


def save_csv_file(df, output_path):
    try:
        df.to_csv(output_path, index=False)
        print(f"已成功将数据保存到 {output_path}")
    except Exception as e:
        print(f"保存文件时发生错误：{e}")


if __name__ == "__main__":
    # 读取带 Is 的表
    is_table_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/Region_CONUSHarvard.csv'
    df_is = read_csv_file(is_table_path)
    if df_is is None:
        exit(1)

    # 指定年份列表
    years = [2011]
    for year in years:
        # 动态构建输入和输出文件路径
        input_table_path = f'/DeepLearning/mnt/shixiansheng/data_fusion/output/Data_OTHER/2011_Other/{year}_HarvardBased_AtF.csv'
        output_path = f'/DeepLearning/mnt/shixiansheng/data_fusion/output/Data_OTHER/2011_Other/{year}_HarvardBased_AtF.csv.csv'

        # 读取输入数据表
        df_input = read_csv_file(input_table_path)
        if df_input is None:
            continue

        # 过滤数据
        df_filtered = filter_data(df_is, df_input)
        if df_filtered is not None:
            # 保存修改后的数据表
            save_csv_file(df_filtered, output_path)
        else:
            print(f"数据过滤过程中出现问题，无法保存 {year} 年的文件。")
    