"""
区域像素边界提取工具

功能：使用 OCR 识别图像中的文本框，提取其像素坐标及相关信息并保存为 JSON 文件
"""
import os
import json
import cv2
from paddleocr import PaddleOCR
from typing import List, Dict, Any


# ==================== 全局配置参数 ====================

# 输入输出路径
image_path = r"D:\myProject\NAN\data\Tile_+028_+014_L22_0005000.jpg"
output_dir = r"D:\myProject\NAN\output"
px_boundry_path = os.path.join(output_dir, "pixel_boundry.json")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 初始化 OCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False)


def calculate_box_info(pts: List[List[float]]) -> Dict[str, Any]:
    """
    根据文本框坐标点计算额外的边界框信息
    
    Args:
        pts: 文本框的 4 个顶点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    
    Returns:
        包含边界框信息的字典
    """
    x_coords = [p[0] for p in pts]
    y_coords = [p[1] for p in pts]
    
    min_x = float(min(x_coords))
    max_x = float(max(x_coords))
    min_y = float(min(y_coords))
    max_y = float(max(y_coords))
    
    # 计算中心点和宽高
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    width = max_x - min_x
    height = max_y - min_y
    
    # 计算边界框（矩形）
    bbox = {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "center": [center_x, center_y],
        "width": width,
        "height": height
    }
    
    return bbox


def extract_px_boundary(image_file: str, output_file: str) -> None:
    """
    从图像中提取文本框的像素边界坐标及相关信息
    
    Args:
        image_file: 输入图像文件路径
        output_file: 输出 JSON 文件路径
    """
    # 加载图像
    image = cv2.imread(image_file)
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_file}")
    
    # OCR 识别
    print("🔍 正在进行 OCR 识别...")
    ocr_result = ocr.ocr(image_file, cls=True)[0]
    
    if not ocr_result:
        print("⚠️  未检测到任何文本")
        return
    
    # 提取文本框信息
    px_boundry = []
    for idx, box_info in enumerate(ocr_result):
        pts = box_info[0]  # 文本框的 4 个顶点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        text = box_info[1][0]  # 识别的文本内容
        confidence = float(box_info[1][1])  # 识别置信度
        
        # 计算边界框信息
        bbox_info = calculate_box_info(pts)
        
        # 构建完整的信息字典
        region_info = {
            "index": idx,  # 区域索引
            "points": pts,  # 4 个顶点坐标（保留原有格式以兼容）
            "text": text,  # 识别的文本内容
            "confidence": confidence,  # 识别置信度 (0-1)
            "bbox": bbox_info  # 边界框信息
        }
        
        px_boundry.append(region_info)
        print(f"✅ [{idx}] 文本: '{text}' | 置信度: {confidence:.3f} | 尺寸: {bbox_info['width']:.1f}x{bbox_info['height']:.1f}")
    
    # 保存为 JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(px_boundry, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ 共提取 {len(px_boundry)} 个文本区域的像素边界")
    print(f"📄 已保存到: {output_file}")


if __name__ == "__main__":
    extract_px_boundary(image_path, px_boundry_path)
