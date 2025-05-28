import os
from PIL import Image

# 定义时间段顺序
periods = ['DJF', 'MAM', 'JJA', 'SON', 'Apr-Sep', 'Annual', 'top-10', 'W126']

# 定义基础路径和输出文件夹
base_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/9ClimateRegion_Timeseries"
output_folder = "/DeepLearning/mnt/shixiansheng/data_fusion/output/9ClimateRegion_Timeseries_W126AtF_Merged"

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
    
    # 计算需要的行数（每行4张图片）
    num_images = len(images)
    num_cols = 4  # 每行4张图片
    num_rows = (num_images + num_cols - 1) // num_cols  # 向上取整计算行数
    
    # 创建新的空白图片用于合并
    merged_width = width * num_cols
    merged_height = height * num_rows
    merged_image = Image.new('RGB', (merged_width, merged_height), color='white')
    
    # 将每张图片粘贴到合并图片的对应位置
    for i, img in enumerate(images):
        row = i // num_cols
        col = i % num_cols
        x = col * width
        y = row * height
        merged_image.paste(img, (x, y))
    
    # 保存合并后的图片，文件名中添加PopWeighted
    output_path = os.path.join(output_folder, f"{region_code}_Merged.png")
    merged_image.save(output_path)
    print(f"  合并图片已保存至: {output_path}")

print("\n所有地区处理完成！")