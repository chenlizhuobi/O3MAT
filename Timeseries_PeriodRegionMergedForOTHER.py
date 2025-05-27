import os
from PIL import Image

# 定义时间段顺序
periods = ['top-10', 'W126', 'W126_AtF']

# 定义基础路径和输出文件夹
base_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/9ClimateRegion_PopWeightedCountyBased_Timeseries"
output_folder = "/DeepLearning/mnt/shixiansheng/data_fusion/output/9ClimateRegion_PopWeightedCountyBased_Timeseries_Merged"

#非人口，覆盖
base_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/9ClimateRegion_Timeseries"
output_folder = "/DeepLearning/mnt/shixiansheng/data_fusion/output/9ClimateRegion_Timeseries_Merged"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义需要处理的地区代码
region_codes = [
    'NE_CR',  # Northeast
    'NRP_CR', # Northern Rockies and Plains
    'NW_CR',  # Northwest
    'CEN_CR', # Ohio Valley
    'S_CR',   # South
    'SE_CR',  # Southeast
    'SW_CR',  # Southwest
    'UPMW_CR',# Upper Midwest
    'USA',    # USA
    'W_CR'
]

# 处理每个地区
for region_code in region_codes:
    print(f"\n正在处理地区: {region_code}")
    images = []
    
    # 收集所有时间段的图片
    for period in periods:
        # 构建图片路径
        if period == 'W126_AtF':
            # AtF情况的图片路径
            image_path = os.path.join(base_path, f"W126_{region_code}_Ozone_PopWeighted_Timeseries_AtF.png")
        else:
            # 普通情况的图片路径
            image_path = os.path.join(base_path, f"{period}_{region_code}_Ozone_PopWeighted_Timeseries.png")

        # 非人口加权
        if period == 'W126_AtF':
            # AtF情况的图片路径
            image_path = os.path.join(base_path, f"W126_{region_code}_Ozone_Timeseries_AtF.png")
        else:
            # 普通情况的图片路径
            image_path = os.path.join(base_path, f"{period}_{region_code}_Ozone_Timeseries.png")
        
        # 检查图片是否存在
        if os.path.exists(image_path):
            img = Image.open(image_path)
            images.append(img)
            print(f"  找到图片: {image_path}")
        else:
            print(f"  未找到图片: {image_path}")
    
    # 如果没有找到任何图片，则跳过此地区
    if not images:
        print(f"  没有找到任何与 {region_code} 相关的图片，跳过此地区。")
        continue
    
    # 获取第一张图片的尺寸作为基准
    width, height = images[0].size
    
    # 创建新的空白图片用于合并
    merged_image = Image.new('RGB', (width * len(images), height), color='white')
    
    # 将每张图片粘贴到合并图片的对应位置
    for i, img in enumerate(images):
        x = i * width
        merged_image.paste(img, (x, 0))
    
    # 保存合并后的图片
    output_path = os.path.join(output_folder, f"{region_code}_Merged_WithAtF.png")
    merged_image.save(output_path)
    print(f"  合并图片已保存至: {output_path}")

print("\n所有地区处理完成！")
