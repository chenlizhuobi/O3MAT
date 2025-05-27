import pandas as pd
import os
import re
from typing import Dict, Tuple, List, Optional
from scipy import stats
import numpy as np

def analyze_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    分析单个CSV文件，返回avna_ozone > 0和<0的子集数据
    """
    try:
        df = pd.read_csv(file_path)
        
        # 检查必要的列是否存在
        required_cols = ['avna_ozone', 'vna_bias', 'model']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in file {file_path}")
                return pd.DataFrame(), pd.DataFrame()
        
        # 筛选出avna_ozone非空的行
        valid_data = df[df['avna_ozone'].notna()].copy()
        
        # 分割为avna_ozone > 0和<0的子集
        positive_subset = valid_data[valid_data['avna_ozone'] > 0][['avna_ozone', 'vna_bias', 'model']]
        negative_subset = valid_data[valid_data['avna_ozone'] < 0][['avna_ozone', 'vna_bias', 'model']]
        
        return positive_subset, negative_subset
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return pd.DataFrame(), pd.DataFrame()

def is_valid_filename(filename: str) -> bool:
    """检查文件名是否符合'YYYY_W126_AtF.csv'格式"""
    return re.match(r'\d{4}_W126_AtF\.csv', filename) is not None

def process_all_files(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """处理目录下所有符合格式的CSV文件，返回合并后的正负子集数据"""
    all_positive = []
    all_negative = []
    
    for filename in os.listdir(data_dir):
        if is_valid_filename(filename):
            file_path = os.path.join(data_dir, filename)
            pos_data, neg_data = analyze_data(file_path)
            
            if not pos_data.empty:
                all_positive.append(pos_data)
                print(f"Processed {filename}: {len(pos_data)} positive rows")
            if not neg_data.empty:
                all_negative.append(neg_data)
                print(f"Processed {filename}: {len(neg_data)} negative rows")
    
    # 合并所有子集
    combined_positive = pd.concat(all_positive, ignore_index=True) if all_positive else pd.DataFrame()
    combined_negative = pd.concat(all_negative, ignore_index=True) if all_negative else pd.DataFrame()
    
    print(f"\nTotal positive rows: {len(combined_positive)}")
    print(f"Total negative rows: {len(combined_negative)}")
    
    return combined_positive, combined_negative

def analyze_correlations(positive_data: pd.DataFrame, negative_data: pd.DataFrame) -> None:
    """分析正负子集中的相关性"""
    # 定义分析函数（避免代码重复）
    def analyze_subset(subset: pd.DataFrame, condition: str):
        if subset.empty:
            print(f"\nNo data for {condition} (avna_ozone {'>0' if condition=='positive' else '<0'})")
            return
        
        print(f"\n===== {condition.upper()} CORRELATION ANALYSIS (avna_ozone {'>0' if condition=='positive' else '<0'}) =====")
        print(f"Data points: {len(subset)}")
        
        # 1. avna_ozone 与 vna_bias 的皮尔逊相关（数值变量）
        if 'vna_bias' in subset.columns:
            try:
                corr_vna, p_vna = stats.pearsonr(subset['avna_ozone'], subset['vna_bias'])
                print(f"\n- avna_ozone vs vna_bias (Pearson):")
                print(f"  Correlation: {corr_vna:.4f}, P-value: {p_vna:.4f}")
                if p_vna < 0.05:
                    print("  相关性显著")
                else:
                    print("  相关性不显著")
                print(f"  强度: {'强' if abs(corr_vna)>=0.7 else '中' if abs(corr_vna)>=0.3 else '弱'}")
            except Exception as e:
                print(f"Error calculating Pearson correlation for {condition}: {e}")
        
        # 2. avna_ozone（分箱）与 model 的卡方检验（分类变量）
        if 'model' in subset.columns:
            try:
                # 对avna_ozone分箱（例如分为3箱）
                subset['avna_bin'] = pd.qcut(subset['avna_ozone'], q=3, labels=['Low', 'Medium', 'High'])
                contingency_table = pd.crosstab(subset['avna_bin'], subset['model'])
                
                print("\n- avna_ozone (分箱) vs model (卡方检验):")
                print("  列联表:")
                print(contingency_table)
                
                chi2, p_chi, dof, _ = stats.chi2_contingency(contingency_table)
                print(f"  Chi-square: {chi2:.4f}, P-value: {p_chi:.4f}")
                if p_chi < 0.05:
                    print("  关联显著")
                else:
                    print("  关联不显著")
                
                # Cramer's V 效应量
                n = contingency_table.sum().sum()
                cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                print(f"  效应量 (Cramer's V): {cramers_v:.4f}")
                print(f"  强度: {'强' if cramers_v>=0.5 else '中' if cramers_v>=0.3 else '弱'}")
            except Exception as e:
                print(f"Error calculating Chi-square test for {condition}: {e}")

    # 分析正数子集
    analyze_subset(positive_data, "positive")
    # 分析负数子集
    analyze_subset(negative_data, "negative")

def main():
    """主函数"""
    data_dir = "/DeepLearning/mnt/shixiansheng/data_fusion/output/W126_AtF"
    print(f"开始处理目录: {data_dir}")
    
    # 处理数据
    positive_data, negative_data = process_all_files(data_dir)
    
    # 分析相关性
    analyze_correlations(positive_data, negative_data)

if __name__ == "__main__":
    main()