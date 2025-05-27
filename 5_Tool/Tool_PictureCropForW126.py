import os
from PIL import Image

def crop_image(input_path, output_path=None, top_percent=0, bottom_percent=0, right_percent=0):
    """
    按百分比裁剪图片的上下右三个方向
    
    参数:
        input_path: 输入图片路径
        output_path: 输出图片路径，默认为None表示覆盖保存
        top_percent: 顶部裁剪百分比(0-100)
        bottom_percent: 底部裁剪百分比(0-100)
        right_percent: 右侧裁剪百分比(0-100)
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"文件不存在: {input_path}")
            
        # 打开图片
        with Image.open(input_path) as img:
            width, height = img.size
            
            # 计算裁剪区域
            left = 0
            top = int(height * top_percent / 100)
            right = width - int(width * right_percent / 100)
            bottom = height - int(height * bottom_percent / 100)
            
            # 验证裁剪区域
            if top >= bottom or left >= right:
                raise ValueError(f"裁剪区域无效: top={top}, bottom={bottom}, left={left}, right={right}")
                
            # 执行裁剪
            cropped_img = img.crop((left, top, right, bottom))
            
            # 确定输出路径
            if output_path is None:
                output_path = input_path
                
            # 保存图片（覆盖原文件）
            cropped_img.save(output_path)
            print(f"✔ 裁剪成功: {os.path.basename(input_path)}")
            return True
            
    except Exception as e:
        print(f"✘ 裁剪失败: {os.path.basename(input_path)} - {str(e)}")
        return False

def process_batch(base_dir, periods, variables, years, crop_params):
    """
    批量处理图片（覆盖保存）
    
    参数:
        base_dir: 基础目录
        periods: 时间段列表
        variables: 变量列表
        years: 年份列表
        crop_params: 裁剪参数
    """
    total_files = len(periods) * len(variables) * len(years)
    processed_files = 0
    success_files = 0
    
    print(f"开始批量处理图片，共{total_files}个文件...")
    
    for year in years:
        year_dir = os.path.join(base_dir, f"{year}_AloneMap")
        
        # 检查年份目录是否存在
        if not os.path.exists(year_dir):
            print(f"警告: 年份目录不存在 - {year_dir}")
            continue
            
        for period in periods:
            # 替换特殊字符，确保目录名合法
            safe_period = period.replace(":", "_").replace(" ", "_")
            
            for variable in variables:
                # 构建文件路径
                file_name = f"{year}_{safe_period}: {variable}.png"
                file_path = os.path.join(year_dir, file_name)
                
                # 裁剪参数
                top_percent = crop_params.get("top_percent", 0)
                bottom_percent = crop_params.get("bottom_percent", 0)
                right_percent = crop_params.get("right_percent", 0)
                
                # 处理单个文件（覆盖保存）
                processed_files += 1
                print(f"\n[{processed_files}/{total_files}] 处理: {file_name}")
                
                if crop_image(file_path, None, top_percent, bottom_percent, right_percent):
                    success_files += 1
    
    print(f"\n批量处理完成！共处理 {processed_files} 个文件，成功 {success_files} 个，失败 {processed_files-success_files} 个。")

if __name__ == "__main__":
    # 基础目录
    BASE_DIR = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/AloneMap_W126"
    
    # 时间段列表
    # PERIODS = ["DJF", "MAM", "JJA", "SON", "Apr-Sep", "Annual", "top-10"]
    PERIODS = ["W126"]
    
    # 变量列表
    VARIABLES = ["EQUATES","VNA", "eVNA", "aVNA"]
    
    # 年份范围
    YEARS = list(range(2002, 2020))  # 2002-2019
    
    # # 裁剪参数（根据需要调整）
    # CROP_PARAMS = {
    #     "top_percent": 25,     # 顶部裁剪25%
    #     "bottom_percent": 19.5,  # 底部裁剪19.5%
    #     "right_percent": 10    # 右侧裁剪10%
    # }
    #对于W126而言
    CROP_PARAMS = {
        "top_percent": 25,     # 顶部裁剪25%
        "bottom_percent": 19.5,  # 底部裁剪19.5%
        "right_percent": 9.5    # 右侧裁剪10%
    }

    # 执行批量处理（覆盖保存）
    process_batch(BASE_DIR, PERIODS, VARIABLES, YEARS, CROP_PARAMS)