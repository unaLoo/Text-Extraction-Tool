"""
区域图像提取与拼接工具

功能：
    - 从原图中提取指定像素坐标区域
    - 对区域进行变换（旋转、缩放、翻转）
    - 根据配置将多个区域拼接为连续图像
"""
import cv2
import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Any


# ==================== 区域提取模块 ====================

def extract_region_from_block(block: List[List[float]], image: np.ndarray, 
                              output_path: str, output_dir: Optional[str] = None) -> Tuple[np.ndarray, str]:
    """
    基于像素坐标从图像中提取一块区域并保存到新文件
    
    Args:
        block: 包含4个点的列表，每个点是一个 [x, y] 坐标
        image: OpenCV 读取的图像 (numpy array)
        output_path: 输出文件的完整路径，或仅文件名（如果提供了 output_dir）
        output_dir: 可选，输出目录（如果 output_path 只是文件名）
    
    Returns:
        Tuple[cropped_image, output_file_path]: 裁剪后的图像和实际保存的文件路径
    
    Raises:
        ValueError: 如果图像为空
    """
    if image is None:
        raise ValueError("图像为空，请检查图像路径是否正确")
    
    # 计算边界框
    x_coords = [int(point[0]) for point in block]
    y_coords = [int(point[1]) for point in block]
    
    min_x = max(0, min(x_coords))
    max_x = min(image.shape[1] - 1, max(x_coords))
    min_y = max(0, min(y_coords))
    max_y = min(image.shape[0] - 1, max(y_coords))
    
    # 裁剪图像区域
    cropped_image = image[min_y:max_y + 1, min_x:max_x + 1]
    
    # 确定输出文件路径并创建目录
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, output_path)
    else:
        output_file_path = output_path
    
    output_dir_path = os.path.dirname(output_file_path)
    if output_dir_path:
        os.makedirs(output_dir_path, exist_ok=True)
    
    # 保存图像
    cv2.imwrite(output_file_path, cropped_image)
    
    print(f"✅ 区域已提取并保存到: {output_file_path}")
    print(f"   裁剪区域: ({min_x}, {min_y}) 到 ({max_x}, {max_y})")
    print(f"   尺寸: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
    
    return cropped_image, output_file_path


def extract_all_regions(px_boundry_file: str, image_file: str, 
                        output_dir: str = "output/extracted_regions") -> List[Dict[str, Any]]:
    """
    从 JSON 文件中读取所有区域坐标，提取所有区域并保存
    
    Args:
        px_boundry_file: 包含像素坐标的 JSON 文件路径
        image_file: 源图像文件路径
        output_dir: 输出目录
    
    Returns:
        extracted_regions: 所有提取的图像列表，每个元素包含 {'index', 'block', 'image', 'output_path'}
    
    Raises:
        ValueError: 如果无法读取图像文件
    """
    # 加载坐标数据和图像
    with open(px_boundry_file, "r", encoding="utf-8") as f:
        px_boundry_data = json.load(f)
    
    image = cv2.imread(image_file)
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_file}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 兼容新旧两种格式
    # 新格式：列表元素是字典，包含 'points' 字段
    # 旧格式：列表元素直接是坐标点列表
    def get_points(item):
        """从数据项中提取坐标点，兼容新旧格式"""
        if isinstance(item, dict) and "points" in item:
            return item["points"]  # 新格式
        elif isinstance(item, list) and len(item) > 0 and isinstance(item[0], list):
            return item  # 旧格式（直接是坐标点列表）
        else:
            raise ValueError(f"无法识别的数据格式: {type(item)}")
    
    # 提取每个区域
    extracted_regions = []
    for i, item in enumerate(px_boundry_data):
        block = get_points(item)  # 提取坐标点
        
        output_filename = f"region_{i}.jpg"
        cropped_img, output_path = extract_region_from_block(
            block, image, output_filename, output_dir
        )
        
        # 如果新格式，保留额外信息
        region_data = {
            "index": i,
            "block": block,
            "image": cropped_img,
            "output_path": output_path
        }
        
        # 如果是新格式，添加额外信息
        if isinstance(item, dict):
            region_data["text"] = item.get("text", "")
            region_data["confidence"] = item.get("confidence", 0.0)
            region_data["bbox"] = item.get("bbox", {})
        
        extracted_regions.append(region_data)
    
    print(f"\n✅ 共提取了 {len(extracted_regions)} 个区域")
    return extracted_regions


# ==================== 图像变换模块 ====================

def apply_transform(image: np.ndarray, transform_config: Dict[str, Any]) -> np.ndarray:
    """
    对图像应用变换（旋转、缩放、翻转）
    
    变换顺序：水平翻转 -> 垂直翻转 -> 旋转 -> 缩放
    
    Args:
        image: 输入图像
        transform_config: 变换配置字典，包含：
            - rotation: 旋转角度（度，顺时针为正）
            - scale: 缩放比例（1.0为原始大小）
            - flip_horizontal: 是否水平翻转
            - flip_vertical: 是否垂直翻转
    
    Returns:
        变换后的图像
    """
    result = image.copy()
    
    # 水平翻转
    if transform_config.get("flip_horizontal", False):
        result = cv2.flip(result, 1)
    
    # 垂直翻转
    if transform_config.get("flip_vertical", False):
        result = cv2.flip(result, 0)
    
    # 旋转
    rotation = transform_config.get("rotation", 0)
    if rotation != 0:
        h, w = result.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
        
        # 计算新的图像尺寸以适应旋转（避免裁剪）
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # 调整旋转中心以适应新尺寸
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]
        
        result = cv2.warpAffine(result, matrix, (new_w, new_h))
    
    # 缩放
    scale = transform_config.get("scale", 1.0)
    if scale != 1.0:
        h, w = result.shape[:2]
        result = cv2.resize(result, (int(w * scale), int(h * scale)))
    
    return result


# ==================== 位置计算模块 ====================

def calculate_position(region_img: np.ndarray, position_config: Dict[str, Any], 
                      placed_regions: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    计算区域在画布上的位置
    
    Args:
        region_img: 区域图像
        position_config: 位置配置字典，包含：
            - type: "anchor" 或 "relative"
            - 对于 anchor: x, y（绝对坐标）
            - 对于 relative: relative_to（参考区域索引）、align（对齐方式）、
                             offset（偏移量）、overlap（重叠像素数）
        placed_regions: 已放置区域的列表，每个元素包含 {'x', 'y', 'w', 'h', 'img'}
    
    Returns:
        (x, y): 位置的坐标
    
    Raises:
        ValueError: 如果位置类型未知或相对区域索引超出范围
    """
    h, w = region_img.shape[:2]
    pos_type = position_config["type"]
    
    if pos_type == "anchor":
        # 锚点定位（通常是第一个区域，作为基准点）
        return position_config.get("x", 0), position_config.get("y", 0)
    
    elif pos_type == "relative":
        # 相对定位
        relative_to_index = position_config.get("relative_to", 0)
        if relative_to_index >= len(placed_regions):
            raise ValueError(f"相对区域索引 {relative_to_index} 超出已放置区域范围")
        
        ref_region = placed_regions[relative_to_index]
        ref_x, ref_y = ref_region["x"], ref_region["y"]
        ref_w, ref_h = ref_region["w"], ref_region["h"]
        
        align = position_config.get("align", "right")
        offset = position_config.get("offset", [0, 0])
        overlap = position_config.get("overlap", 0)  # 负数表示间距
        
        # 根据对齐方式计算位置
        align_map = {
            "right": (ref_x + ref_w - overlap, ref_y + offset[1]),
            "left": (ref_x - w + overlap, ref_y + offset[1]),
            "bottom": (ref_x + offset[0], ref_y + ref_h - overlap),
            "top": (ref_x + offset[0], ref_y - h + overlap),
            "center": (ref_x + (ref_w - w) // 2 + offset[0], 
                      ref_y + (ref_h - h) // 2 + offset[1])
        }
        
        return align_map.get(align, (ref_x + offset[0], ref_y + offset[1]))
    
    else:
        raise ValueError(f"未知的位置类型: {pos_type}")


def calculate_canvas_size(processed_regions: List[Dict[str, Any]], 
                          output_config: Dict[str, Any]) -> Tuple[int, int]:
    """
    计算画布尺寸
    
    Args:
        processed_regions: 已处理的区域列表（包含变换后的图像和位置配置）
        output_config: 输出配置，包含 size 和 background_color
    
    Returns:
        (width, height): 画布的宽度和高度
    """
    # 临时计算所有区域的位置，以确定画布大小
    temp_placed = []
    min_x = min_y = 0
    max_x = max_y = 0
    
    for proc_region in processed_regions:
        x, y = calculate_position(
            proc_region["image"], 
            proc_region["position_config"], 
            temp_placed
        )
        h, w = proc_region["image"].shape[:2]
        temp_placed.append({"x": x, "y": y, "w": w, "h": h})
        
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
    
    canvas_width = max_x - min_x
    canvas_height = max_y - min_y
    
    # 根据配置确定最终尺寸
    size_config = output_config.get("size", "auto")
    if size_config == "auto":
        return canvas_width, canvas_height
    elif isinstance(size_config, list) and len(size_config) >= 2:
        return size_config[0], size_config[1]
    else:
        return canvas_width, canvas_height


def create_canvas(width: int, height: int, bg_color: List[int], 
                  is_rgb: bool = True) -> np.ndarray:
    """
    创建画布
    
    Args:
        width: 画布宽度
        height: 画布高度
        bg_color: 背景颜色 [R, G, B] 或 [gray]
        is_rgb: 是否为RGB图像
    
    Returns:
        创建的画布（numpy array）
    """
    if is_rgb:
        return np.full((height, width, 3), bg_color, dtype=np.uint8)
    else:
        return np.full((height, width), bg_color[0], dtype=np.uint8)


# ==================== 区域合并模块 ====================

def process_regions_for_merge(regions_list: List[Dict[str, Any]], 
                              merge_order: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    处理区域列表，应用变换并准备位置配置
    
    Args:
        regions_list: 区域图像列表，按索引顺序
        merge_order: 合并顺序配置列表
    
    Returns:
        processed_regions: 处理后的区域列表
    
    Raises:
        ValueError: 如果区域索引超出范围或配置无效
    """
    if not merge_order:
        raise ValueError("合并配置中 merge_order 为空")
    
    processed_regions = []
    
    for item in merge_order:
        region_idx = item["region_index"]
        if region_idx >= len(regions_list):
            raise ValueError(f"区域索引 {region_idx} 超出范围（共 {len(regions_list)} 个区域）")
        
        # 获取原始图像并应用变换
        region_img = regions_list[region_idx]["image"]
        transform = item.get("transform", {})
        transformed_img = apply_transform(region_img, transform)
        
        processed_regions.append({
            "index": region_idx,
            "name": item.get("name", f"Region_{region_idx}"),
            "image": transformed_img,
            "transform": transform,
            "position_config": item.get("position", {})
        })
    
    return processed_regions


def place_region_on_canvas(canvas: np.ndarray, region_img: np.ndarray, 
                           x: int, y: int, min_offset: Tuple[int, int]) -> None:
    """
    将区域放置在画布上（处理超出边界的情况）
    
    Args:
        canvas: 画布图像
        region_img: 要放置的区域图像
        x, y: 目标位置（相对于原始坐标系）
        min_offset: (min_x, min_y) 偏移量，用于坐标转换
    """
    # 转换为画布坐标系
    canvas_x = x - min_offset[0]
    canvas_y = y - min_offset[1]
    
    h, w = region_img.shape[:2]
    canvas_h, canvas_w = canvas.shape[:2]
    
    # 计算实际放置区域（处理超出边界的情况）
    src_x = max(0, -canvas_x)
    src_y = max(0, -canvas_y)
    src_w = min(w, canvas_w - max(0, canvas_x)) - src_x
    src_h = min(h, canvas_h - max(0, canvas_y)) - src_y
    
    place_x = max(0, canvas_x)
    place_y = max(0, canvas_y)
    
    if src_w > 0 and src_h > 0:
        canvas[place_y:place_y + src_h, place_x:place_x + src_w] = \
            region_img[src_y:src_y + src_h, src_x:src_x + src_w]


def merge_regions(regions_list: List[Dict[str, Any]], merge_config_file: str, 
                  output_path: str) -> np.ndarray:
    """
    根据配置合并多个区域
    
    Args:
        regions_list: 区域图像列表，按索引顺序
        merge_config_file: 合并配置 JSON 文件路径
        output_path: 输出图像路径
    
    Returns:
        合并后的图像（numpy array）
    
    Raises:
        ValueError: 如果配置无效或区域索引超出范围
        FileNotFoundError: 如果配置文件不存在
    """
    # 加载配置
    with open(merge_config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    merge_order = config.get("merge_order", [])
    output_config = config.get("output", {})
    
    # 处理区域：应用变换并准备位置配置
    processed_regions = process_regions_for_merge(regions_list, merge_order)
    
    # 计算画布尺寸
    canvas_width, canvas_height = calculate_canvas_size(processed_regions, output_config)
    
    # 重新计算位置以获取 min_x, min_y（用于坐标转换）
    temp_placed = []
    min_x = min_y = 0
    
    for proc_region in processed_regions:
        x, y = calculate_position(
            proc_region["image"], 
            proc_region["position_config"], 
            temp_placed
        )
        h, w = proc_region["image"].shape[:2]
        temp_placed.append({"x": x, "y": y, "w": w, "h": h})
        min_x = min(min_x, x)
        min_y = min(min_y, y)
    
    # 创建画布
    bg_color = output_config.get("background_color", [255, 255, 255])
    is_rgb = len(processed_regions) > 0 and len(processed_regions[0]["image"].shape) == 3
    canvas = create_canvas(canvas_width, canvas_height, bg_color, is_rgb)
    
    # 放置每个区域到画布上
    placed_regions = []
    min_offset = (min_x, min_y)
    
    for proc_region in processed_regions:
        x, y = calculate_position(
            proc_region["image"], 
            proc_region["position_config"], 
            placed_regions
        )
        
        region_img = proc_region["image"]
        h, w = region_img.shape[:2]
        
        # 放置图像
        place_region_on_canvas(canvas, region_img, x, y, min_offset)
        
        # 记录已放置区域信息（用于后续相对定位）
        placed_regions.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "img": region_img
        })
        
        print(f"✅ 放置区域 {proc_region['name']} 到位置 ({x - min_x}, {y - min_y})")
    
    # 保存结果
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(output_path, canvas)
    print(f"\n✅ 合并完成，保存到: {output_path}")
    print(f"   画布尺寸: {canvas_width}x{canvas_height}")
    
    return canvas


if __name__ == "__main__":
    # 配置路径
    texture_fp = r"D:\myProject\NAN\demoData\Tile_+028_+014_L22_0005000.jpg"
    px_boundry_fp = r"D:\myProject\NAN\demoData\pixel_boundry.json"
    merge_config_fp = r"D:\myProject\NAN\demoData\merge_config_auto.json"
    
    # 提取所有区域
    print("\n" + "=" * 50)
    print("步骤1: 提取所有区域")
    print("=" * 50)
    all_regions = extract_all_regions(
        px_boundry_fp,
        texture_fp,
        "output/extracted_regions"
    )
    
    # 根据配置合并区域
    print("\n" + "=" * 50)
    print("步骤2: 根据配置合并区域")
    print("=" * 50)
    merged_image = merge_regions(
        all_regions,
        merge_config_fp,
        "output/extracted_regions/$$$$merged_result.jpg"
    )