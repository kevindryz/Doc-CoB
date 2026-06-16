import json
from PIL import Image, ImageDraw
import numpy as np
import os
import uuid
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import random
random.seed(42)

from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = 933120000
import os, glob
import uuid
import math
import re
import argparse

from box_2_id import read_jsonl_file, save_to_jsonl


def mask_image(image_path, coordinates_list, save_folder):
    # 打开图片
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size

    # 获取原始文件名和后缀
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)

    # 创建存储文件夹如果没有的话
    os.makedirs(save_folder, exist_ok=True)
    
    # 创建绘图对象
    draw = ImageDraw.Draw(img)
    
    # 计算框的宽度，假设为图像较短边的0.002
    min_side = min(width, height)
    outline_width = max(1, int(min_side * 0.002))  # 框宽度为图像较短边的0.2%
    
    # 在图片上画多个红框 (框的颜色和宽度可以调整)
    outline_color = (255, 0, 0)  # 红色
    for bbox in coordinates_list:
        draw.rectangle(bbox, outline=outline_color, width=outline_width)
    
    # 生成唯一后缀
    unique_suffix = str(uuid.uuid4())  # 生成唯一后缀
    new_file_name = f"{unique_suffix}_{name}{ext}"
    save_path = os.path.join(save_folder, new_file_name)
    
    # 保存新的图片
    if '.jpg' in save_path:
        img=img.convert('RGB')
    img.save(save_path)
    
    return save_path


def process_item(item, new_image_save_path):
    if "draw_box" in item:
        image_path = item["image_path"].replace('workspace', 'NAS_SHARE')
        box = item["draw_box"]
        new_image_path = mask_image(image_path, box, new_image_save_path)
        item["image_path"] = new_image_path
        item["image"] = new_image_path

        # 定义正则表达式来匹配<img>标签内部的路径
        # pattern = r'(<img>)(.*?)(</img>)'
        # # 使用re.sub替换路径
        # item["conversations"][0]["value"] = re.sub(pattern, r'\1' + new_image_path + r'\3', item["conversations"][0]["value"])
    return item

def process_data(new_image_root, input_path, output_path):
    # 加载数据
    data = read_jsonl_file(input_path)
    # with open(input_path, 'r', encoding='utf-8') as f:
    #     data = json.load(f)

    # 设置图片保存路径
    new_image_save_path = os.path.join(new_image_root)

    # 检查并创建保存路径
    if not os.path.exists(new_image_save_path):
        os.makedirs(new_image_save_path)

    # 使用 process_map 进行数据处理
    data = process_map(process_item, data, [new_image_save_path] * len(data), max_workers=os.cpu_count())

    # 保存处理后的数据
    # with open(output_path, "w", encoding='utf-8') as f:
    #     json.dump(data, f, indent=2, ensure_ascii=False)
    save_to_jsonl(data, output_path)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")

    # 根目录
    base_pat=os.path.join(data_dir, "s1")

    pat_lis=glob.glob(f'{base_pat}/todraw/*')

    # pat_lis=glob.glob(f'{base_pat}/Intermediate_results_s1_draw/*')
    for filee in pat_lis:
        # 创建解析器
        parser = argparse.ArgumentParser(description="Process some data.")
        # 添加命令行参数
        
        parser.add_argument('--new_image_root',type=str, required=False, default=f'{base_pat}/images', help='Path to the new image root directory.')
        parser.add_argument('--input_path', type=str, required=False, default=filee, help='Path to the input JSON file.')
        parser.add_argument('--output_path', type=str, required=False, default=filee, help='Path to save the output JSON file.')

        # 解析命令行参数
        args = parser.parse_args()

        # 调用函数并传递参数
        process_data(args.new_image_root, args.input_path, args.output_path)
            