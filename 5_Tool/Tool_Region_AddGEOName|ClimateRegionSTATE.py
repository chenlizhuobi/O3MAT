# import pandas as pd

# # 读取数据表
# df = pd.read_csv('/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/2011_299*459_9ClimateRegions.csv')

# # 定义地区名称映射字典
# region_mapping = {
#     0: 'Northeast',
#     1: 'North Rockies and Plains',
#     2: 'Northwest',
#     3: 'Ohio Valley',
#     4: 'South',
#     5: 'Southeast',
#     6: 'Southwest',
#     7: 'Upper Midwest',
#     8: 'West'
# }

# # 根据ClimateRegion中的id获取对应的地区名称，并创建新列ClimateRegionName
# df['ClimateRegionName'] = df['ClimateRegion'].map(region_mapping)

# # 覆盖保存原文件（不包含行索引）
# df.to_csv('/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/2011_299*459_9ClimateRegions.csv', index=False)

import pandas as pd
import chardet
import os
# 气候区与州名的映射关系（基于之前的表格数据）
climate_region_mapping = {
    'Northeast': [
        'Connecticut', 'Delaware', 'Maine', 'Maryland', 
        'Massachusetts', 'New Hampshire', 'New Jersey', 
        'New York', 'Pennsylvania', 'Rhode Island', 'Vermont'
    ],
    'Upper Midwest': [
        'Iowa', 'Michigan', 'Minnesota', 'Wisconsin'
    ],
    'Ohio Valley': [
        'Illinois', 'Indiana', 'Kentucky', 'Missouri', 
        'Ohio', 'Tennessee', 'West Virginia'
    ],
    'Southeast': [
        'Alabama', 'Florida', 'Georgia', 'North Carolina', 
        'South Carolina', 'Virginia'
    ],
    'Northern Rockies and Plains': [
        'Montana', 'Nebraska', 'North Dakota', 'South Dakota', 
        'Wyoming'
    ],
    'South': [
        'Arkansas', 'Kansas', 'Louisiana', 'Mississippi', 
        'Oklahoma', 'Texas'
    ],
    'Southwest': [
        'Arizona', 'Colorado', 'New Mexico', 'Utah'
    ],
    'Northwest': [
        'Idaho', 'Oregon', 'Washington'
    ],
    'West': [
        'California', 'Nevada'
    ]
}

def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        # 读取前1MB数据进行编码检测
        raw_data = f.read(1024 * 1024)
        result = chardet.detect(raw_data)
    return result['encoding']

def read_csv_with_encoding(file_path):
    """尝试多种编码读取CSV文件"""
    # 常见编码列表
    common_encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252', 'utf-8-sig', 'gbk']
    
    # 先尝试自动检测编码
    detected_encoding = detect_encoding(file_path)
    print(f"自动检测到的编码: {detected_encoding}")
    
    # 构建尝试列表（优先使用检测到的编码）
    encodings_to_try = [detected_encoding] + [e for e in common_encodings if e != detected_encoding]
    
    for encoding in encodings_to_try:
        try:
            print(f"尝试使用编码: {encoding}")
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"✅ 成功使用编码: {encoding}")
            return df
        except UnicodeDecodeError as e:
            print(f"❌ 编码 {encoding} 失败: {str(e)}")
        except Exception as e:
            print(f"❌ 读取文件时发生未知错误: {str(e)}")
    
    # 如果所有编码都失败，尝试使用错误替换模式
    print("⚠️ 所有标准编码均失败，尝试使用错误替换模式...")
    try:
        df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
        print("✅ 已使用错误替换模式读取文件（可能包含替换字符）")
        return df
    except Exception as e:
        print(f"❌ 无法读取文件: {str(e)}")
        return None

def main():
    # 文件路径
    file_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/UnitedStatesCensusCountyPopulation2010to2020.csv'
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误：文件不存在 - {file_path}")
        return
    
    # 读取数据
    census_df = read_csv_with_encoding(file_path)
    if census_df is None:
        print("❌ 无法读取数据，程序终止")
        return
    
    # 提取所有气候区包含的州名（去重）
    all_climate_states = set()
    for states in climate_region_mapping.values():
        all_climate_states.update(states)
    
    # 检查STATE_NAME列是否存在
    if 'STATE_NAME' not in census_df.columns:
        print(f"❌ 数据集中未找到 'STATE_NAME' 列。可用列名: {list(census_df.columns)}")
        return
    
    # 获取所有州名（去重）
    census_states = set(census_df['STATE_NAME'].unique())
    
    # 找出不在气候区映射中的州
    missing_states = census_states - all_climate_states
    
    if not missing_states:
        print("✅ 所有人口普查数据中的州均被气候区包含！")
    else:
        print(f"❗ 发现 {len(missing_states)} 个未被气候区包含的州:")
        for state in sorted(missing_states):
            print(f"  - {state}")
    
    # 统计每个州出现的次数
    state_counts = census_df['STATE_NAME'].value_counts()
    print("\n各州数据记录数量统计:")
    print(state_counts)
    
    # 分析气候区覆盖情况
    region_coverage = {}
    for region, states in climate_region_mapping.items():
        region_states = set(states)
        covered = region_states.intersection(census_states)
        coverage_percent = len(covered) / len(region_states) * 100
        region_coverage[region] = {
            'total_states': len(region_states),
            'covered_states': len(covered),
            'coverage_percentage': coverage_percent
        }
    
    # 打印气候区覆盖情况
    print("\n气候区覆盖情况:")
    for region, stats in sorted(region_coverage.items(), key=lambda x: x[1]['coverage_percentage'], reverse=True):
        print(f"{region}: {stats['covered_states']}/{stats['total_states']} ({stats['coverage_percentage']:.1f}%)")

if __name__ == "__main__":
    main()