import os
import json
from pyproj import CRS, Transformer
from typing import List, Dict

def convert_json_coordinates(json_file_path: str) -> Dict:
    """
    转换单个JSON文件的坐标，应用所有变换参数
    :param json_file_path: JSON文件路径
    :return: 转换后的JSON数据
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # 获取所有变换参数
    hc_transform = data.get('hc-transform', {}).get('default', {})
    proj_str = hc_transform.get('crs', '')
    scale = hc_transform.get('scale')
    jsonmarginX = hc_transform.get('jsonmarginX', 0)
    jsonmarginY = hc_transform.get('jsonmarginY', 0)
    xoffset = hc_transform.get('xoffset', 0)
    yoffset = hc_transform.get('yoffset', 0)
    
    # 参数校验
    if not proj_str:
        raise ValueError(f"文件 {json_file_path} 中缺少投影参数 (hc-transform.default.crs)")
    if scale is None:
        raise ValueError(f"文件 {json_file_path} 中缺少缩放因子 (hc-transform.default.scale)")
    
    # 创建坐标系转换器
    src_crs = CRS.from_proj4(proj_str)
    dst_crs = CRS("EPSG:4326")  # WGS84
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    
    # 遍历所有几何对象
    for feature in data['features']:
        geometry_type = feature['geometry']['type']
        coordinates = feature['geometry']['coordinates']
        
        # 应用所有变换参数进行坐标转换
        if geometry_type == "Polygon":
            converted_coords = convert_polygon(
                coordinates, transformer, scale,
                jsonmarginX, jsonmarginY, xoffset, yoffset
            )
        elif geometry_type == "MultiPolygon":
            converted_coords = convert_multi_polygon(
                coordinates, transformer, scale,
                jsonmarginX, jsonmarginY, xoffset, yoffset
            )
        else:
            raise NotImplementedError(f"不支持的几何类型: {geometry_type}")
        
        feature['geometry']['coordinates'] = converted_coords
    
    return data

def convert_polygon(
    coords: List[List[List[float]]],
    transformer: Transformer,
    scale: float,
    jsonmarginX: float,
    jsonmarginY: float,
    xoffset: float,
    yoffset: float
) -> List[List[List[float]]]:
    """转换Polygon坐标，应用所有变换参数（修正坐标计算逻辑）"""
    converted = []
    for ring in coords:
        xys = []
        for x, y in ring:
            # 1. 修正：从存储坐标中扣除jsonmargin获取原始坐标
            x_actual = x - jsonmarginX
            y_actual = y - jsonmarginY
            
            # 2. 应用缩放因子（转换为投影坐标单位）
            x_scaled = x_actual * scale
            y_scaled = y_actual * scale
            
            # 3. 应用offset（投影坐标偏移）
            x_proj = x_scaled + xoffset
            y_proj = y_scaled + yoffset
            
            # 调试输出（可在生产环境中注释掉）
            print(f"原始存储坐标: ({x}, {y})")
            print(f"扣除jsonmargin后: ({x_actual}, {y_actual})")
            print(f"缩放并应用offset后: ({x_proj:.2f}, {y_proj:.2f})")
            
            xys.append((x_proj, y_proj))
        
        # 4. 坐标转换到WGS84
        lons, lats = transformer.transform(*zip(*xys))
        converted_ring = [[round(lon, 6), round(lat, 6)] for lon, lat in zip(lons, lats)]
        print(f"转换为经纬度: {converted_ring[0]} (示例点)")
        converted.append(converted_ring)
    
    return converted

def convert_multi_polygon(
    coords: List[List[List[List[float]]]],
    transformer: Transformer,
    scale: float,
    jsonmarginX: float,
    jsonmarginY: float,
    xoffset: float,
    yoffset: float
) -> List[List[List[List[float]]]]:
    """转换MultiPolygon坐标"""
    converted = []
    for polygon in coords:
        converted_polygon = convert_polygon(
            polygon, transformer, scale,
            jsonmarginX, jsonmarginY, xoffset, yoffset
        )
        converted.append(converted_polygon)
    return converted

def batch_convert_json_files(input_dir: str, output_dir: str):
    """批量转换文件夹下的所有JSON文件（保持不变）"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            try:
                converted_data = convert_json_coordinates(file_path)
                output_path = os.path.join(output_dir, f"converted_{filename}")
                with open(output_path, 'w') as f:
                    json.dump(converted_data, f, indent=2)
                print(f"成功转换文件: {filename}")
            except Exception as e:
                print(f"转换文件 {filename} 失败: {str(e)}")

if __name__ == "__main__":
    input_directory = "/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/Region(STATEIncluedCOUNTY)"
    output_directory = "/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/Region(STATEIncluedCOUNTY)Transformed"
    batch_convert_json_files(input_directory, output_directory)

# import csv
# import json
# import os
# from collections import defaultdict
# import logging

# # 配置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler()]
# )

# def clean_county_name(county_name):
#     """清洗县名，去除常见行政后缀"""
#     if not isinstance(county_name, str):
#         return None
        
#     # 去除常见后缀（如County、Parish等）
#     suffixes = [' county', ' parish', ' borough', ' city', ' township', ' district', ' county,', ' parish,']
#     county_lower = county_name.lower()
    
#     for suffix in suffixes:
#         if county_lower.endswith(suffix):
#             return county_name[:-len(suffix)].strip()
    
#     return county_name

# def get_county_data_from_csv(csv_path):
#     """从CSV中提取州名和县名（已清洗）"""
#     state_field = 'STATE_NAME'
#     county_field = 'COUNTY_NAME'
#     state_county_map = defaultdict(set)
    
#     with open(csv_path, 'r', encoding='utf-8') as f:
#         reader = csv.DictReader(f)
#         if state_field not in reader.fieldnames or county_field not in reader.fieldnames:
#             logging.error(f"CSV缺少必要字段！需要{state_field}和{county_field}")
#             return {}
        
#         for row in reader:
#             state = row[state_field].strip()
#             county = clean_county_name(row[county_field])
#             if state and county:
#                 state_county_map[state].add(county)
    
#     logging.info(f"从CSV中提取{len(state_county_map)}个州的{sum(len(c) for c in state_county_map.values())}个县")
#     return state_county_map

# def get_json_path_by_state(state_name, json_folder):
#     """根据州名查找对应的JSON文件路径（精确匹配文件名）"""
#     expected_filename = f"converted_{state_name}.json"
#     json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    
#     if expected_filename in json_files:
#         return os.path.join(json_folder, expected_filename)
#     for filename in json_files:
#         if filename.lower() == expected_filename.lower():
#             logging.warning(f"州名{state_name}大小写不完全匹配，但找到文件: {filename}")
#             return os.path.join(json_folder, filename)
#     logging.warning(f"未找到州{state_name}对应的JSON文件（期望文件名: {expected_filename}）")
#     return None

# def extract_counties_from_json(json_path, target_counties):
#     """提取县名及其完整多边形数据（严格遵循GeoJSON规范）"""
#     matched = defaultdict(dict)
#     try:
#         with open(json_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         if 'features' not in data or not isinstance(data['features'], list):
#             logging.warning(f"{json_path}缺少'features'数组，格式异常")
#             return matched
        
#         for feature in data['features']:
#             props = feature.get('properties', {})
#             name = props.get('NAME') or props.get('name') or props.get('COUNTY_NAME') or props.get('CountyName')
#             cleaned_name = clean_county_name(name)
            
#             if cleaned_name in target_counties:
#                 geometry = feature.get('geometry', {})
#                 geom_type = geometry.get('type')
#                 coordinates = geometry.get('coordinates', [])
                
#                 # 保存完整几何对象（支持Polygon和MultiPolygon）
#                 if geom_type in ['Polygon', 'MultiPolygon']:
#                     matched[cleaned_name] = {
#                         'geometry': geometry,  # 完整几何对象
#                         'properties': {
#                             'county_name': cleaned_name,
#                             'state': None  # 后续填充州名
#                         }
#                     }
#                 else:
#                     logging.warning(f"{cleaned_name}的几何类型{geom_type}不受支持")
    
#     except Exception as e:
#         logging.error(f"处理{json_path}失败: {str(e)}")
#     return matched

# def main():
#     csv_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/UnitedStatesCensusCountyPopulation2010to2020.csv'
#     json_folder = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/Region(STATEIncluedCOUNTY)Transformed'
#     output_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/USA_CONUTYS.json'
#     diff_path = 'county_diff_report.txt'
    
#     state_county_map = get_county_data_from_csv(csv_path)
#     if not state_county_map:
#         return
    
#     matched = defaultdict(list)  # {州名: [几何对象]}
#     missing = defaultdict(set)
#     next_id = 1
    
#     for state, target_counties in state_county_map.items():
#         json_path = get_json_path_by_state(state, json_folder)
#         if not json_path:
#             missing[state].update(target_counties)
#             continue
        
#         county_data = extract_counties_from_json(json_path, target_counties)
#         for county, data in county_data.items():
#             data['properties']['state'] = state
#             data['properties']['id'] = next_id
#             matched[state].append(data)
#             next_id += 1
        
#         matched_counties = set(county_data.keys())
#         missing[state].update(target_counties - matched_counties)
    
#     # 整理为标准GeoJSON格式
#     geojson_output = {
#         "type": "FeatureCollection",
#         "features": [
#             {
#                 "type": "Feature",
#                 "geometry": item['geometry'],
#                 "properties": item['properties']
#             }
#             for state_counties in matched.values()
#             for item in state_counties
#         ]
#     }
    
#     # 输出统计信息
#     total_matched = sum(len(features) for features in matched.values())
#     total_target = sum(len(counties) for counties in state_county_map.values())
#     total_missing = total_target - total_matched  # 计算未匹配数
    
#     logging.info("\n===== 匹配结果汇总 =====")
#     logging.info(f"总目标县数: {total_target}")
#     logging.info(f"成功匹配数: {total_matched} ({total_matched/total_target*100:.2f}%)")
#     logging.info(f"分配的ID范围: 1-{next_id-1}")
    
#     # 保存结果
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(geojson_output, f, ensure_ascii=False, indent=2)
    
#     # 生成差异报告（同前）
#     with open(diff_path, 'w', encoding='utf-8') as f:
#         f.write("===== 县名匹配差异报告 =====\n")
#         f.write(f"CSV路径: {csv_path}\n")
#         f.write(f"JSON文件夹路径: {json_folder}\n\n")
#         f.write(f"总目标县数: {total_target}\n")
#         f.write(f"成功匹配数: {total_matched} ({total_matched/total_target*100:.2f}%)\n")
#         f.write(f"未匹配数: {total_missing} ({total_missing/total_target*100:.2f}%)\n\n")
        
#         if missing:
#             f.write("按州列出未匹配的县:\n")
#             for state, counties in sorted(missing.items(), key=lambda x: len(x[1]), reverse=True):
#                 f.write(f"\n{state} ({len(counties)}个):\n")
#                 for county in sorted(counties):
#                     f.write(f"- {county}\n")

# if __name__ == "__main__":
#     main()
# import json
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon
# from matplotlib.collections import PatchCollection
# import numpy as np

# def plot_single_json(json_file_path: str, title: str = None, figsize: tuple = (10, 10)) -> None:
#     """
#     绘制单个JSON文件中的地理数据
    
#     Args:
#         json_file_path: JSON文件路径
#         title: 图表标题，默认为文件名
#         figsize: 图像大小
#     """
#     # 设置中文字体支持
#     plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    
#     # 创建图形和轴
#     plt.figure(figsize=figsize)
#     ax = plt.axes()
    
#     # 如果没有提供标题，使用文件名
#     if not title:
#         title = json_file_path.split("/")[-1].replace(".json", "")
    
#     try:
#         # 读取JSON文件
#         with open(json_file_path, 'r') as f:
#             data = json.load(f)
        
#         # 收集所有多边形补丁和边界范围
#         patches = []
#         colors = []
#         min_lon, max_lon = float('inf'), -float('inf')
#         min_lat, max_lat = float('inf'), -float('inf')
        
#         # 处理每个特征（县/区域）
#         for feature in data.get('features', []):
#             geometry = feature.get('geometry')
#             if not geometry:
#                 continue
                
#             geometry_type = geometry.get('type')
#             coordinates = geometry.get('coordinates')
            
#             # 获取区域名称（如果有）
#             properties = feature.get('properties', {})
#             name = properties.get('name', '未知')
            
#             # 为不同的区域生成随机颜色
#             color = np.random.rand(3,)
            
#             if geometry_type == 'Polygon':
#                 # 处理单个多边形
#                 for ring in coordinates:
#                     # 验证坐标格式
#                     valid_ring = []
#                     for point in ring:
#                         if isinstance(point, list) and len(point) >= 2:
#                             valid_ring.append([point[0], point[1]])  # 确保是[x,y]格式
                    
#                     if len(valid_ring) < 3:  # 至少需要3个点来形成多边形
#                         continue
                    
#                     # 更新边界范围
#                     lon = [point[0] for point in valid_ring]
#                     lat = [point[1] for point in valid_ring]
#                     min_lon = min(min_lon, min(lon))
#                     max_lon = max(max_lon, max(lon))
#                     min_lat = min(min_lat, min(lat))
#                     max_lat = max(max_lat, max(lat))
                    
#                     # 创建多边形
#                     polygon = Polygon(valid_ring, closed=True)
#                     patches.append(polygon)
#                     colors.append(color)
                    
#             elif geometry_type == 'MultiPolygon':
#                 # 处理多个多边形
#                 for polygon_coords in coordinates:
#                     for ring in polygon_coords:
#                         # 验证坐标格式
#                         valid_ring = []
#                         for point in ring:
#                             if isinstance(point, list) and len(point) >= 2:
#                                 valid_ring.append([point[0], point[1]])  # 确保是[x,y]格式
                        
#                         if len(valid_ring) < 3:  # 至少需要3个点来形成多边形
#                             continue
                        
#                         # 更新边界范围
#                         lon = [point[0] for point in valid_ring]
#                         lat = [point[1] for point in valid_ring]
#                         min_lon = min(min_lon, min(lon))
#                         max_lon = max(max_lon, max(lon))
#                         min_lat = min(min_lat, min(lat))
#                         max_lat = max(max_lat, max(lat))
                        
#                         # 创建多边形
#                         polygon = Polygon(valid_ring, closed=True)
#                         patches.append(polygon)
#                         colors.append(color)
        
#         # 检查是否有补丁要绘制
#         if not patches:
#             print("没有找到可绘制的几何特征")
#             return
        
#         # 创建补丁集合
#         p = PatchCollection(patches, alpha=0.6)
#         p.set_color(colors)
#         ax.add_collection(p)
        
#         # 设置坐标轴范围（添加一些边距）
#         margin_lon = (max_lon - min_lon) * 0.05
#         margin_lat = (max_lat - min_lat) * 0.05
#         ax.set_xlim(min_lon - margin_lon, max_lon + margin_lon)
#         ax.set_ylim(min_lat - margin_lat, max_lat + margin_lat)
        
#         # 设置纵横比为1:1，确保经纬度比例正确
#         ax.set_aspect('equal')
        
#         # 添加网格线
#         ax.grid(True, linestyle='--', alpha=0.7)
        
#         # 添加标题和标签
#         plt.title(title, fontsize=16)
#         plt.xlabel('经度', fontsize=12)
#         plt.ylabel('纬度', fontsize=12)
        
#         # 显示地图
#         plt.tight_layout()
#         plt.show()
        
#     except Exception as e:
#         print(f"处理文件时出错: {str(e)}")

# if __name__ == "__main__":
#     # 指定要绘制的JSON文件路径
#     json_file = "/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/Region(STATEIncluedCOUNTY)Transformed/converted_Texas.json"
    
#     # 绘制单个JSON文件
#     plot_single_json(json_file, title="德克萨斯州地图")