import json
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import shape
import numpy as np
import os

# 定义气候区数据
climate_regions = {
    '001': {'id': 0, 'name': 'Northeast', 'cmaq_name': 'NE_CR'},
    '002': {'id': 1, 'name': 'Northern Rockies and Plains', 'cmaq_name': 'NRP_CR'},
    '003': {'id': 2, 'name': 'Northwest', 'cmaq_name': 'NW_CR'},
    '004': {'id': 3, 'name': 'Ohio Valley', 'cmaq_name': 'CEN_CR'},
    '005': {'id': 4, 'name': 'South', 'cmaq_name': 'S_CR'},
    '006': {'id': 5, 'name': 'Southeast', 'cmaq_name': 'SE_CR'},
    '007': {'id': 6, 'name': 'Southwest', 'cmaq_name': 'SW_CR'},
    '008': {'id': 7, 'name': 'Upper Midwest', 'cmaq_name': 'UPMW_CR'},
    '009': {'id': 8, 'name': 'West', 'cmaq_name': 'W_CR'},
    'USA': {'id': 9, 'name': 'USA', 'cmaq_name': 'USA'}
}

# 读取GeoJSON文件
input_file = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/US_climate_regions.json'
output_dir = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/Region_PythonProcess'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取GeoJSON数据
with open(input_file, 'r') as f:
    data = json.load(f)

# 创建GeoDataFrame
features = []
for feature in data['features']:
    # 获取区域ID
    region_id = feature['id']
    
    # 尝试从properties中获取区域ID（如果id字段不是区域ID）
    if 'properties' in feature and 'region_id' in feature['properties']:
        region_id = feature['properties']['region_id']
    
    # 获取区域名称
    region_name = climate_regions.get(region_id, {}).get('name', f"Region {region_id}")
    
    # 创建几何形状
    geometry = shape(feature['geometry'])
    
    # 添加到特征列表
    features.append({
        'id': region_id,
        'name': region_name,
        'geometry': geometry
    })

# 创建GeoDataFrame
gdf = gpd.GeoDataFrame(features)

# 创建颜色映射
colors = plt.cm.tab10(np.linspace(0, 1, len(gdf)))

# 创建图形
plt.figure(figsize=(12, 8))

# 绘制每个区域
for i, (idx, row) in enumerate(gdf.iterrows()):
    # 检查是否为Southeast区域
    is_southeast = row['name'] == 'Southeast'
    
    # 为Southeast区域使用不同的颜色或样式
    if is_southeast:
        color = 'red'  # 或者其他明显的颜色
        edgecolor = 'black'
        linewidth = 2
    else:
        color = colors[i % len(colors)]
        edgecolor = 'gray'
        linewidth = 1
    
    # 绘制区域
    gpd.GeoSeries(row['geometry']).plot(
        ax=plt.gca(), 
        color=color, 
        edgecolor=edgecolor, 
        linewidth=linewidth
    )
    
    # 计算区域的中心点用于标注
    centroid = row['geometry'].centroid
    
    # 标注区域名称
    plt.text(centroid.x, centroid.y, row['name'], 
             fontsize=8, ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

# 设置标题和样式
plt.title('US Climate Regions')
plt.axis('off')  # 关闭坐标轴

# 保存图像
output_file = os.path.join(output_dir, 'US_climate_regions_map.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"地图已保存到: {output_file}")
