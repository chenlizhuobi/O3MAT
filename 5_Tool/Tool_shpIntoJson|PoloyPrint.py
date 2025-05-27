import geopandas as gpd
import json

def convert_shp_to_json(input_path, output_path):
    """
    将shapefile文件转换为JSON格式
    
    参数:
    input_path (str): 输入shapefile文件路径
    output_path (str): 输出JSON文件路径
    """
    try:
        # 读取shapefile文件
        gdf = gpd.read_file(input_path)
        
        # 确保数据包含geometry列
        if 'geometry' not in gdf.columns:
            raise ValueError("输入的shapefile不包含geometry列")
        
        # 将GeoDataFrame转换为JSON格式
        json_data = gdf.to_json()
        
        # 解析JSON字符串为Python对象
        data = json.loads(json_data)
        
        # 保存为JSON文件
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"成功将shapefile转换为JSON并保存至 {output_path}")
        
    except Exception as e:
        print(f"转换过程中出现错误: {e}")

# 使用示例
if __name__ == "__main__":
    input_file = "/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/CountyRegion/cb_2020_us_county_500k.shp"  # 替换为实际的shapefile文件路径
    output_file = "/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/CountyRegion/cb_2020_us_county_500k.json"  # 替换为期望的输出路径
    
    convert_shp_to_json(input_file, output_file)

