import os
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon
import pyproj
import json
import pyrsig  # 用于读取ioapi格式的模型文件

def read_json(file_path):
    """读取JSON文件并返回特征列表和坐标列表"""
    with open(file_path, "r") as f:
        data = json.load(f)
    
    features = data.get("features", [])  # 获取所有feature
    return features  # 只返回特征列表，坐标处理移至generate_polygons_within_usa函数

def get_projection_from_model(model_file):
    """从模型文件中获取投影信息"""
    ds_model = pyrsig.open_ioapi(model_file)
    return ds_model.crs_proj4

def generate_polygons_within_usa(model_file, json_path):
    """定义美国的多个县区域，并从模型文件获取投影信息"""
    features = read_json(json_path)
    
    # 统计区域数量（根据features数组长度）
    region_count = len(features)
    print(f"检测到 {region_count} 个县区域（通过JSON的features数量确定）")
    
    # 从模型文件获取投影信息
    proj_string = get_projection_from_model(model_file)
    
    # 创建投影转换器
    proj = pyproj.CRS.from_proj4(proj_string)
    transformer = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), proj, always_xy=True)

    # 将每个坐标集合转换为单独的多边形，并关联其ID、州名和县名
    polygons = []
    for feature in features:
        # 从properties中获取县名称和州名称
        properties = feature.get("properties", {})
        county_name = properties.get("NAME", "未知县")
        state_name = properties.get("STATE_NAME", "未知州")
        
        # 从properties中获取GEOID作为唯一标识
        geoid = properties.get("GEOID", "0")
        
        # 确保ID有效
        if not geoid:
            print(f"警告: 县 {county_name}, {state_name} 缺少GEOID，将被忽略")
            continue
            
        # 将ID转换为整数（如果是字符串形式的数字）
        try:
            state_id = int(geoid)
        except ValueError:
            print(f"警告: 县 {county_name}, {state_name} 的GEOID {geoid} 无法转换为整数，将被忽略")
            continue
            
        geom_type = feature["geometry"]["type"]
        coords = feature["geometry"]["coordinates"]
        
        try:
            if geom_type == "Polygon":
                # 处理单个多边形
                exterior_coords = coords[0]  # 外环坐标
                interior_coords = coords[1:] if len(coords) > 1 else []  # 内环坐标（如果有）
                
                # 转换坐标
                exterior_transformed = [transformer.transform(lon, lat) for lon, lat in exterior_coords]
                interiors_transformed = [
                    [transformer.transform(lon, lat) for lon, lat in ring] 
                    for ring in interior_coords
                ]
                
                # 创建带有孔洞的多边形
                poly = Polygon(exterior_transformed, interiors_transformed)
                polygons.append((state_id, state_name, county_name, poly))
                
            elif geom_type == "MultiPolygon":
                # 处理多个多边形
                sub_polygons = []
                for poly_coords in coords:
                    exterior_coords = poly_coords[0]  # 外环坐标
                    interior_coords = poly_coords[1:] if len(poly_coords) > 1 else []  # 内环坐标（如果有）
                    
                    # 转换坐标
                    exterior_transformed = [transformer.transform(lon, lat) for lon, lat in exterior_coords]
                    interiors_transformed = [
                        [transformer.transform(lon, lat) for lon, lat in ring] 
                        for ring in interior_coords
                    ]
                    
                    # 创建带有孔洞的多边形
                    sub_poly = Polygon(exterior_transformed, interiors_transformed)
                    sub_polygons.append(sub_poly)
                
                # 创建MultiPolygon对象
                multi_poly = MultiPolygon(sub_polygons)
                polygons.append((state_id, state_name, county_name, multi_poly))
                
            else:
                print(f"警告: 不支持的几何类型 {geom_type}，将被忽略")
                
        except Exception as e:
            print(f"警告: 处理县 {county_name}, {state_name} (ID: {state_id}) 时出错: {str(e)}，将被忽略")
    
    return polygons

def filter_points_with_multiple_polygons(df, save_filtered_file, polygons):
    """使用多个县多边形进行过滤，并设置对应的StateRegion值和StateRegionId"""
    # 初始化StateRegion为-999（不在任何多边形内）
    df['StateRegionId'] = -999
    df['StateRegion'] = "-999"  # 直接标注为字符串"-999"
    df['StateName'] = "-999"    # 初始化州名
    df['CountyName'] = "-999"  # 初始化县名
    
    def get_state_region(row, polygons):
        point = Point(row["COL"], row["ROW"])
        for state_id, state_name, county_name, poly in polygons:
            if poly.contains(point):
                return (state_id, state_name, county_name)
        return (-999, "-999", "-999")  # 不在任何多边形内，返回"-999"
    
    # 同时获取县ID、州名和县名
    df['StateRegionId'], df['StateName'], df['CountyName'] = zip(*df.apply(lambda row: get_state_region(row, polygons), axis=1))
    
    # 添加Period列并设为None
    df['Period'] = "None"
    
    result_df = df[['ROW', 'COL', 'StateRegionId', 'StateName', 'CountyName', 'Period']]  # 包含新列
    result_df.to_csv(save_filtered_file, index=False)
    
    # 统计每个州的点数
    state_counts = result_df['StateName'].value_counts().sort_index()
    print("\n各州的点数统计:")
    for state_name, count in state_counts.items():
        print(f"{state_name}: {count} 个点")
    
    # 统计每个县的点数（取前10个）
    county_counts = result_df['CountyName'].value_counts().sort_values(ascending=False).head(10)
    print("\n点数最多的10个县:")
    for county_name, count in county_counts.items():
        print(f"{county_name}: {count} 个点")
    
    print(f"\n已保存结果至 {save_filtered_file}")

if __name__ == "__main__":
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/Region"
    os.makedirs(save_path, exist_ok=True)
    
    model_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/HR2DAY_LST_ACONC_v532_cb6r3_ae7_aq_WR413_MYR_STAGE_2011_12US1_2011.nc"
    json_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/CountyRegion/cb_2020_us_county_500k.json"  # 使用县JSON文件
    data_fusion_file = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/HourlyData_WithoutCV/2011_W126_ST_Limit.csv"
    filtered_file = os.path.join(save_path, "2011_299*459_CountyRegions.csv")  # 输出文件名改为CountyRegions
    
    # 读取JSON并统计区域数量
    features = read_json(json_path)
    print(f"JSON文件中包含 {len(features)} 个县区域（通过features数组长度确定）")
    
    # 生成多边形（包含从feature.properties获取的真实ID、州名和县名）
    usa_polygons = generate_polygons_within_usa(model_file, json_path)
    
    # 过滤数据
    df_data = pd.read_csv(data_fusion_file)
    filter_points_with_multiple_polygons(df_data, filtered_file, usa_polygons)
    print("处理完成!")    