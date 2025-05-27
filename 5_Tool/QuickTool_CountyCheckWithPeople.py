import pandas as pd

# 定义文件路径（示例路径，需替换为实际路径）
input_file = "/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/2011_299*459_CountyRegions.csv"
census_file = "/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/UnitedStatesCensusCountyPopulation2010to2020.csv"
region_not_in_census_output = "/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/RegionCounties_NotInCensus.csv"

def process_county_data():
    try:
        # 读取并预处理区域数据
        df = pd.read_csv(input_file)
        df['CountyName'] = df['CountyName'].astype(str)  # 统一转为字符串
        valid_df = df[df['CountyName'] != '-999']  # 剔除CountyName为-999的记录
        
        # 读取人口普查数据并构建匹配字典
        census_df = pd.read_csv(census_file)
        census_mapping = set(
            (state.strip().lower(), county.strip().lower()) 
            for state, county in zip(census_df['STATE_NAME'], census_df['COUNTY_NAME'])
            if pd.notna(state) and pd.notna(county)
        )
        
        # 标记匹配状态
        valid_df['InCensus'] = valid_df.apply(
            lambda row: (
                row['StateName'].strip().lower(), 
                row['CountyName'].strip().lower()
            ) in census_mapping,
            axis=1
        )
        
        # 提取未匹配数据
        unmatched_df = valid_df[~valid_df['InCensus']][['StateName', 'CountyName']]
        unmatched_count = len(unmatched_df)
        
        # 输出统计结果
        print(f"\n区域数据匹配结果:")
        print(f"有效区域数据行数: {len(valid_df)}")
        print(f"匹配成功行数: {len(valid_df) - unmatched_count}")
        print(f"未匹配行数: {unmatched_count}")
        
        if unmatched_count > 0:
            # 打印所有未匹配的州和县（支持大数据量分页显示）
            print("\n未匹配的州和县列表（共{}条）:".format(unmatched_count))
            
            # 分批次打印（每50条一批，避免控制台输出过长）
            batch_size = 50
            for i in range(0, unmatched_count, batch_size):
                batch = unmatched_df.iloc[i:i+batch_size]
                print(batch)
                
                # 非最后一批时，添加分隔线
                if i + batch_size < unmatched_count:
                    print("-" * 50)
            
            # 保存到文件
            unmatched_df.to_csv(region_not_in_census_output, index=False)
            print(f"\n完整未匹配数据已保存至: {region_not_in_census_output}")
        else:
            print("\n所有有效数据均匹配成功！")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    process_county_data()