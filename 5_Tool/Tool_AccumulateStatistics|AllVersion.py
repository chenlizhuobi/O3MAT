import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from typing import Dict, Tuple, List, Optional

# 设置字体为Times New Roman（罗马字体）
plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def analyze_data(file_path: str) -> Tuple[int, int, List[float]]:
    """
    Analyze a single CSV file to calculate:
    1. Proportion of rows where avna_ozone < 0
    2. Average model for rows where avna_ozone < 0
    
    Args:
    file_path (str): Path to the CSV file
    
    Returns:
    Tuple[int, int, List[float]]: 
        (Number of rows where avna_ozone < 0, Total valid rows, List of model values where avna_ozone < 0)
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_cols = ['avna_ozone', 'model']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in file {file_path}")
                return 0, 0, []
        
        # Filter out rows where avna_ozone is NaN
        valid_data = df[df['avna_ozone'].notna()]
        
        # Count rows where avna_ozone < 0 and get corresponding model values
        negative_data = valid_data[valid_data['avna_ozone'] < 0]
        negative_count = len(negative_data)
        model_values = negative_data['model'].dropna().tolist()
        
        # Total valid rows
        total_valid = len(valid_data)
        
        return negative_count, total_valid, model_values
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return 0, 0, []

def get_year_from_filename(filename: str) -> int:
    """
    Extract the year from the filename
    
    Args:
    filename (str): Filename
    
    Returns:
    int: Extracted year, or 0 if extraction fails
    """
    # Assume filename format is "YYYY_Wxxx_AtF.csv"
    match = re.search(r'(\d{4})_W\d+_AtF\.csv', filename)
    if match:
        return int(match.group(1))
    return 0

def process_all_files(data_dir: str, start_year: int = 2002, end_year: int = 2019) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Process all CSV files in the specified directory and calculate:
    1. Proportion of rows where avna_ozone < 0 for each year
    2. Average model for rows where avna_ozone < 0 for each year
    
    Args:
    data_dir (str): Data directory path
    start_year (int): Start year
    end_year (int): End year
    
    Returns:
    Tuple[Dict[int, float], Dict[int, float]]: 
        (Mapping from year to percentage of avna_ozone < 0, 
         Mapping from year to average model where avna_ozone < 0)
    """
    year_results = {year: (0, 0, []) for year in range(start_year, end_year + 1)}
    
    # Iterate over all files in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            year = get_year_from_filename(filename)
            if start_year <= year <= end_year:
                file_path = os.path.join(data_dir, filename)
                neg_count, total, model_values = analyze_data(file_path)
                
                # Accumulate results
                current_neg, current_total, current_bias_list = year_results.get(year, (0, 0, []))
                year_results[year] = (current_neg + neg_count, current_total + total, current_bias_list + model_values)
    
    # Calculate percentages and average model
    year_percentages = {}
    year_avg_bias = {}
    
    for year, (neg, total, bias_list) in year_results.items():
        # Calculate percentage of avna_ozone < 0
        if total > 0:
            percentage = (neg / total) * 100
            year_percentages[year] = percentage
            print(f"{year}: Percentage of avna_ozone < 0 = {percentage:.2f}% ({neg}/{total})")
        else:
            year_percentages[year] = 0
            print(f"{year}: No valid data for percentage calculation")
        
        # Calculate average model where avna_ozone < 0
        if bias_list:
            avg_bias = sum(bias_list) / len(bias_list)
            year_avg_bias[year] = avg_bias
            print(f"{year}: Average model for avna_ozone < 0 = {avg_bias:.4f} ({len(bias_list)} samples)")
        else:
            year_avg_bias[year] = 0
            print(f"{year}: No valid model data for avna_ozone < 0")
    
    return year_percentages, year_avg_bias

def plot_results(year_percentages: Dict[int, float], year_avg_bias: Dict[int, float], output_dir: str = "/DeepLearning/mnt/shixiansheng/data_fusion/output/TestMap"):
    """
    Plot two bar charts:
    1. Percentage of Rows with avna_ozone < 0
    2. Average model for rows where avna_ozone < 0
    
    Args:
    year_percentages (Dict[int, float]): Mapping from year to percentage
    year_avg_bias (Dict[int, float]): Mapping from year to average model
    output_dir (str): Directory to save the charts
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract years and corresponding values
    years = sorted(year_percentages.keys())
    percentages = [year_percentages[year] for year in years]
    avg_biases = [year_avg_bias[year] for year in years]
    
    # ----------------------
    # Plot 1: Percentage Chart
    # ----------------------
    plt.figure(figsize=(12, 6))
    bars = plt.bar(years, percentages, color='skyblue', edgecolor='black')
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}%', ha='center', va='bottom', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1.0))
    
    # Set chart title and axis labels
    plt.title('Percentage of Rows with avna_ozone < 0 (2002 - 2019)')
    plt.xlabel('Year')
    plt.ylabel('Percentage (%)')
    plt.xticks(years)
    plt.ylim(0, max(percentages) * 1.1 if percentages else 10)  # Set y-axis range
    
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the chart
    output_path1 = os.path.join(output_dir, "avna_ozone_percentage.png")
    plt.tight_layout()
    plt.savefig(output_path1)
    print(f"Chart saved to: {output_path1}")
    
    # ----------------------
    # Plot 2: Average VNA Bias Chart
    # ----------------------
    plt.figure(figsize=(12, 6))
    # Use red for negative values and blue for positive values
    colors = ['red' if bias < 0 else 'blue' for bias in avg_biases]
    bars = plt.bar(years, avg_biases, color=colors, edgecolor='black')
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}', ha='center', 
                 va='top' if height < 0 else 'bottom', 
                 fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1.0))
    
    # Set chart title and axis labels
    plt.title('Average model for Rows with avna_ozone < 0 (2002 - 2019)')
    plt.xlabel('Year')
    plt.ylabel('Average model')
    plt.xticks(years)
    
    # Ensure y-axis includes zero and provides some padding
    min_val = min(avg_biases) if avg_biases else -1
    max_val = max(avg_biases) if avg_biases else 1
    padding = max(abs(min_val), abs(max_val)) * 0.1
    plt.ylim(min(min_val - padding, 0), max(max_val + padding, 0))
    
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the chart
    output_path2 = os.path.join(output_dir, "model_average.png")
    plt.tight_layout()
    plt.savefig(output_path2)
    print(f"Chart saved to: {output_path2}")
    
    # Display the charts
    plt.show()

def main():
    """Main function"""
    # Data directory path
    data_dir = "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF"
    output_dir = "/DeepLearning/mnt/shixiansheng/data_fusion/output/TestMap"
    
    # Process data
    year_percentages, year_avg_bias = process_all_files(data_dir)
    
    # Plot charts
    plot_results(year_percentages, year_avg_bias, output_dir)

if __name__ == "__main__":
    main()    

# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import re
# from typing import Dict, Tuple, List, Optional

# plt.rcParams["font.family"] = ["Times New Roman"]
# plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# def analyze_data(file_path: str) -> Tuple[
#     int, int,  # 负样本数, 总有效数
#     List[float], List[float],  # 负样本 model, vna_bias
#     List[float], List[float]   # 正样本 model, vna_bias
# ]:
#     try:
#         df = pd.read_csv(file_path)
#         required_cols = ['avna_ozone', 'model', 'vna_bias']
#         for col in required_cols:
#             if col not in df.columns:
#                 print(f"Warning: Column '{col}' not found in file {file_path}")
#                 return 0, 0, [], [], [], []
        
#         valid_data = df[df['avna_ozone'].notna()]
#         negative_data = valid_data[valid_data['avna_ozone'] < 0]
#         positive_data = valid_data[valid_data['avna_ozone'] > 0]
        
#         # 负样本数据
#         neg_count = len(negative_data)
#         model_neg = negative_data['model'].dropna().tolist()
#         vna_neg = negative_data['vna_bias'].dropna().tolist()
        
#         # 正样本数据
#         pos_count = len(positive_data)
#         model_pos = positive_data['model'].dropna().tolist()
#         vna_pos = positive_data['vna_bias'].dropna().tolist()
        
#         return neg_count, len(valid_data), model_neg, vna_neg, model_pos, vna_pos
    
#     except Exception as e:
#         print(f"Error processing file {file_path}: {e}")
#         return 0, 0, [], [], [], []

# def get_year_from_filename(filename: str) -> int:
#     match = re.search(r'(\d{4})_W\d+_AtF\.csv', filename)
#     return int(match.group(1)) if match else 0

# def process_all_files(data_dir: str, start_year: int = 2002, end_year: int = 2019) -> Tuple[
#     Dict[int, float], Dict[int, float],  # 负样本 model/vna_bias 均值
#     Dict[int, float], Dict[int, float]   # 正样本 model/vna_bias 均值
# ]:
#     year_results = {
#         year: (0, 0, [], [], [], [])  # (neg_count, total, model_neg, vna_neg, model_pos, vna_pos)
#         for year in range(start_year, end_year + 1)
#     }
    
#     for filename in os.listdir(data_dir):
#         if filename.endswith('.csv'):
#             year = get_year_from_filename(filename)
#             if start_year <= year <= end_year:
#                 file_path = os.path.join(data_dir, filename)
#                 neg_cnt, total, mod_neg, vna_neg, mod_pos, vna_pos = analyze_data(file_path)
                
#                 yr_data = year_results[year]
#                 year_results[year] = (
#                     yr_data[0] + neg_cnt,
#                     yr_data[1] + total,
#                     yr_data[2] + mod_neg,
#                     yr_data[3] + vna_neg,
#                     yr_data[4] + mod_pos,
#                     yr_data[5] + vna_pos
#                 )
    
#     # 计算各类均值，保留两位小数
#     year_avg_mod_neg = {}
#     year_avg_vna_neg = {}
#     year_avg_mod_pos = {}
#     year_avg_vna_pos = {}
    
#     for year, (neg_cnt, total, mod_neg, vna_neg, mod_pos, vna_pos) in year_results.items():
#         year_avg_mod_neg[year] = round(sum(mod_neg)/len(mod_neg), 2) if mod_neg else 0.0
#         year_avg_vna_neg[year] = round(sum(vna_neg)/len(vna_neg), 2) if vna_neg else 0.0
#         year_avg_mod_pos[year] = round(sum(mod_pos)/len(mod_pos), 2) if mod_pos else 0.0
#         year_avg_vna_pos[year] = round(sum(vna_pos)/len(vna_pos), 2) if vna_pos else 0.0
    
#     return year_avg_mod_neg, year_avg_vna_neg, year_avg_mod_pos, year_avg_vna_pos

# def plot_results(
#     year_avg_mod_neg: Dict[int, float], year_avg_vna_neg: Dict[int, float],
#     year_avg_mod_pos: Dict[int, float], year_avg_vna_pos: Dict[int, float],
#     output_dir: str = "/DeepLearning/mnt/shixiansheng/data_fusion/output/TestMap"
# ):
#     os.makedirs(output_dir, exist_ok=True)
#     years = sorted(year_avg_mod_neg.keys())
#     n = len(years)
#     width = 0.2  # 单个柱形宽度
    
#     fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    
#     # 绘制四组柱形
#     x_neg_mod = [i - 1.5*width for i in range(n)]
#     bars_neg_mod = ax.bar(x_neg_mod, year_avg_mod_neg.values(), width, 
#                          color='#1f77b4', edgecolor='black', label='Model (aVNA<0)')
    
#     x_neg_vna = [i - 0.5*width for i in range(n)]
#     bars_neg_vna = ax.bar(x_neg_vna, year_avg_vna_neg.values(), width, 
#                          color='#ff7f0e', edgecolor='black', label='VNA Bias (aVNA<0)')
    
#     x_pos_mod = [i + 0.5*width for i in range(n)]
#     bars_pos_mod = ax.bar(x_pos_mod, year_avg_mod_pos.values(), width, 
#                          color='#2ca02c', edgecolor='black', label='Model (aVNA>0)')
    
#     x_pos_vna = [i + 1.5*width for i in range(n)]
#     bars_pos_vna = ax.bar(x_pos_vna, year_avg_vna_pos.values(), width, 
#                          color='#d62728', edgecolor='black', label='VNA Bias (aVNA>0)')
    
#     # 设置坐标轴和标题
#     ax.set_xlabel('Year', fontsize=12)
#     ax.set_ylabel('Average Value', fontsize=12)
#     ax.set_title('Combined Analysis of Ozone Data by Year (2002-2019)', fontsize=16)
    
#     # 设置刻度标签
#     ax.set_xticks(range(n))
#     ax.set_xticklabels(years, rotation=45, ha='right')
    
#     # 确保y轴包含0并合理扩展
#     all_values = [
#         *year_avg_mod_neg.values(), *year_avg_vna_neg.values(),
#         *year_avg_mod_pos.values(), *year_avg_vna_pos.values()
#     ]
#     if all_values:
#         min_val = min(min(all_values), 0) * 1.1
#         max_val = max(max(all_values), 0) * 1.1
#         ax.set_ylim(min_val, max_val)
    
#     # 添加数据标签
#     def add_labels(bars, ax):
#         for bar in bars:
#             height = bar.get_height()
#             text = f"{height:.2f}"
#             ax.text(bar.get_x() + bar.get_width()/2, height, text, 
#                    ha='center', va='bottom' if height > 0 else 'top', 
#                    fontsize=9, fontweight='bold')
    
#     add_labels(bars_neg_mod, ax)
#     add_labels(bars_neg_vna, ax)
#     add_labels(bars_pos_mod, ax)
#     add_labels(bars_pos_vna, ax)
    
#     # 添加图例和基线
#     ax.legend(loc='upper right', ncol=2)
#     ax.axhline(0, color='black', linestyle='-', linewidth=1.2)
    
#     # 网格线
#     ax.grid(axis='y', linestyle='--', alpha=0.7)
    
#     # 保存图片
#     output_path = os.path.join(output_dir, "model_vna_bias_comparison.png")
#     plt.tight_layout()
#     plt.savefig(output_path)
#     print(f"Chart saved to: {output_path}")
#     plt.show()

# def main():
#     data_dir = "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF"
#     output_dir = "/DeepLearning/mnt/shixiansheng/data_fusion/output/TestMap"
    
#     mod_neg, vna_neg, mod_pos, vna_pos = process_all_files(data_dir)
#     plot_results(mod_neg, vna_neg, mod_pos, vna_pos, output_dir)

# if __name__ == "__main__":
#     main()