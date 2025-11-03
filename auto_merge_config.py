"""
自动生成拼接配置文件工具

基于 OCR 结果、UV 信息和 global_index 映射，自动生成 merge_config.json
"""
import os
import json
import numpy as np
import cv2
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import defaultdict


# ==================== 几何计算工具 ====================

def point_in_triangle(pt: Tuple[float, float], v1: Tuple[float, float], 
                     v2: Tuple[float, float], v3: Tuple[float, float]) -> bool:
    """
    判断点是否在三角形内（使用重心坐标法）
    
    Args:
        pt: 待判断的点 (u, v)
        v1, v2, v3: 三角形的三个顶点 (u, v)
    
    Returns:
        True 如果点在三角形内或边上
    """
    u, v = pt
    u1, v1_coord = v1
    u2, v2_coord = v2
    u3, v3_coord = v3
    
    # 计算重心坐标
    denom = (v2_coord - v3_coord) * (u1 - u3) + (u3 - u2) * (v1_coord - v3_coord)
    if abs(denom) < 1e-10:
        return False
    
    a = ((v2_coord - v3_coord) * (u - u3) + (u3 - u2) * (v - v3_coord)) / denom
    b = ((v3_coord - v1_coord) * (u - u3) + (u1 - u3) * (v - v3_coord)) / denom
    c = 1 - a - b
    
    # 点在三角形内当且仅当三个重心坐标都在 [0, 1] 范围内
    return a >= -1e-6 and b >= -1e-6 and c >= -1e-6


def segment_intersects_segment(p1: Tuple[float, float], p2: Tuple[float, float],
                               p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
    """
    判断两条线段是否相交（简化版，使用跨立实验）
    """
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    def on_segment(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))
    
    d1 = cross_product(p3, p4, p1)
    d2 = cross_product(p3, p4, p2)
    d3 = cross_product(p1, p2, p3)
    d4 = cross_product(p1, p2, p4)
    
    # 检查是否相交
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    
    # 检查共线情况
    if d1 == 0 and on_segment(p3, p1, p4):
        return True
    if d2 == 0 and on_segment(p3, p2, p4):
        return True
    if d3 == 0 and on_segment(p1, p3, p2):
        return True
    if d4 == 0 and on_segment(p1, p4, p2):
        return True
    
    return False


def bbox_intersects_triangle(bbox: Dict[str, float], 
                           v1: Tuple[float, float], v2: Tuple[float, float], 
                           v3: Tuple[float, float]) -> bool:
    """
    判断边界框是否与三角形相交
    
    Args:
        bbox: 边界框 {"min_x", "min_y", "max_x", "max_y"} (UV空间)
        v1, v2, v3: 三角形的三个顶点
    
    Returns:
        True 如果相交
    """
    # 检查三角形的包围盒是否与bbox相交
    tri_min_u = min(v1[0], v2[0], v3[0])
    tri_max_u = max(v1[0], v2[0], v3[0])
    tri_min_v = min(v1[1], v2[1], v3[1])
    tri_max_v = max(v1[1], v2[1], v3[1])
    
    # 快速拒绝：包围盒不相交
    if (tri_max_u < bbox["min_x"] or tri_min_u > bbox["max_x"] or
        tri_max_v < bbox["min_y"] or tri_min_v > bbox["max_y"]):
        return False
    
    # 检查bbox的四个角点是否在三角形内
    corners = [
        (bbox["min_x"], bbox["min_y"]),
        (bbox["max_x"], bbox["min_y"]),
        (bbox["max_x"], bbox["max_y"]),
        (bbox["min_x"], bbox["max_y"])
    ]
    
    for corner in corners:
        if point_in_triangle(corner, v1, v2, v3):
            return True
    
    # 检查三角形的顶点是否在bbox内
    for vertex in [v1, v2, v3]:
        if (bbox["min_x"] <= vertex[0] <= bbox["max_x"] and
            bbox["min_y"] <= vertex[1] <= bbox["max_y"]):
            return True
    
    # 检查矩形的边是否与三角形的边相交
    rect_edges = [
        (corners[0], corners[1]),  # 底边
        (corners[1], corners[2]),  # 右边
        (corners[2], corners[3]),  # 顶边
        (corners[3], corners[0])   # 左边
    ]
    
    tri_edges = [
        (v1, v2),
        (v2, v3),
        (v3, v1)
    ]
    
    for rect_edge in rect_edges:
        for tri_edge in tri_edges:
            if segment_intersects_segment(rect_edge[0], rect_edge[1], 
                                         tri_edge[0], tri_edge[1]):
                return True
    
    # 如果包围盒相交但所有检查都没找到相交，返回False
    # （虽然理论上可能存在完全包围但不相交的情况，但实际中很少见）
    return False


# ==================== 坐标转换 ====================

def pixel_to_uv(pixel_coords: List[List[float]], texture_size: List[int]) -> List[Tuple[float, float]]:
    """
    将像素坐标转换为UV坐标
    
    Args:
        pixel_coords: 像素坐标列表 [[x1,y1], [x2,y2], ...]
        texture_size: 纹理尺寸 [width, height]
    
    Returns:
        UV坐标列表 [(u1,v1), (u2,v2), ...]
    """
    w, h = texture_size
    uv_coords = []
    for px, py in pixel_coords:
        u = px / w
        v = 1.0 - py / h  # OpenGL风格，v=0在顶部
        uv_coords.append((u, v))
    return uv_coords


def pixel_bbox_to_uv_bbox(pixel_bbox: Dict[str, float], texture_size: List[int]) -> Dict[str, float]:
    """
    将像素空间的边界框转换为UV空间的边界框
    
    Args:
        pixel_bbox: 像素边界框 {"min_x", "min_y", "max_x", "max_y"}
        texture_size: 纹理尺寸 [width, height]
    
    Returns:
        UV边界框
    """
    w, h = texture_size
    return {
        "min_x": pixel_bbox["min_x"] / w,
        "max_x": pixel_bbox["max_x"] / w,
        "min_y": 1.0 - pixel_bbox["max_y"] / h,  # 注意v坐标翻转
        "max_y": 1.0 - pixel_bbox["min_y"] / h
    }


# ==================== 区域-三角面映射 ====================

def find_regions_faces(ocr_regions: List[Dict[str, Any]], 
                      uv_info: List[Dict[str, Any]]) -> Dict[int, Set[Tuple[int, int]]]:
    """
    为每个OCR区域找到对应的三角面索引
    
    Args:
        ocr_regions: OCR识别结果，每个元素包含 points 和 bbox
        uv_info: UV信息，包含 blocks，每个block有 faces 和 texture_size
    
    Returns:
        region_to_faces: {region_index: {(block_idx, face_id), ...}}
    """
    region_to_faces = defaultdict(set)
    
    
    for block_idx, block in enumerate(uv_info):
        texture_size = block.get("texture_size", [1024, 1024])
        faces = block.get("faces", [])
        
        for region_idx, region in enumerate(ocr_regions):
            # 转换区域边界框到UV空间
            uv_bbox = pixel_bbox_to_uv_bbox(region["bbox"], texture_size)
            
            # 检查每个三角面
            for face in faces:
                uvs = face.get("uvs", [])
                if len(uvs) != 3:
                    continue
                
                v1 = tuple(uvs[0])
                v2 = tuple(uvs[1])
                v3 = tuple(uvs[2])
                
                # 判断边界框是否与三角形相交
                if bbox_intersects_triangle(uv_bbox, v1, v2, v3):
                    face_id = face.get("face_id", 0)
                    region_to_faces[region_idx].add((block_idx, face_id))
    
    return region_to_faces


def get_region_global_indices(region_to_faces: Dict[int, Set[Tuple[int, int]]],
                              uv_info: List[Dict[str, Any]]) -> Dict[int, Set[int]]:
    """
    获取每个区域对应的global_indices集合
    
    Args:
        region_to_faces: {region_index: {(block_idx, face_id), ...}}
        uv_info: UV信息列表
    
    Returns:
        region_to_global_indices: {region_index: {global_index, ...}}
    """
    region_to_global_indices = defaultdict(set)
    
    for region_idx, face_set in region_to_faces.items():
        for block_idx, face_id in face_set:
            block = uv_info[block_idx]
            faces = block.get("faces", [])
            
            # 找到对应的face
            for face in faces:
                if face.get("face_id") == face_id:
                    global_indices = face.get("global_indices", [])
                    region_to_global_indices[region_idx].update(global_indices)
                    break
    
    return region_to_global_indices


# ==================== 邻接关系构建 ====================

def build_adjacency_graph(region_to_global_indices: Dict[int, Set[int]]) -> Dict[int, List[int]]:
    """
    通过shared global_indices构建区域邻接图
    
    Args:
        region_to_global_indices: {region_index: {global_index, ...}}
    
    Returns:
        adjacency_graph: {region_index: [neighbor_region_index, ...]}
    """
    adjacency = defaultdict(list)
    region_list = list(region_to_global_indices.keys())
    
    for i, region_i in enumerate(region_list):
        indices_i = region_to_global_indices[region_i]
        
        for j, region_j in enumerate(region_list[i+1:], i+1):
            indices_j = region_to_global_indices[region_j]
            
            # 如果有共享的global_index，则相邻
            if indices_i & indices_j:  # 集合交集
                adjacency[region_i].append(region_j)
                adjacency[region_j].append(region_i)
    
    return dict(adjacency)


def build_fallback_adjacency_by_position(ocr_regions: List[Dict[str, Any]], 
                                         distance_threshold: float = 100.0) -> Dict[int, List[int]]:
    """
    基于像素位置构建备用邻接图（当无法通过global_index建立邻接时使用）
    
    Args:
        ocr_regions: OCR识别结果
        distance_threshold: 距离阈值，小于此值的区域被认为是相邻的
    
    Returns:
        adjacency_graph: {region_index: [neighbor_region_index, ...]}
    """
    adjacency = defaultdict(list)
    
    for i in range(len(ocr_regions)):
        bbox_i = ocr_regions[i].get("bbox", {})
        center_i = bbox_i.get("center", [0, 0])
        if not isinstance(center_i, list) or len(center_i) < 2:
            continue
        
        for j in range(i + 1, len(ocr_regions)):
            bbox_j = ocr_regions[j].get("bbox", {})
            center_j = bbox_j.get("center", [0, 0])
            if not isinstance(center_j, list) or len(center_j) < 2:
                continue
            
            # 计算欧氏距离
            dx = center_j[0] - center_i[0]
            dy = center_j[1] - center_i[1]
            distance = np.sqrt(dx * dx + dy * dy)
            
            # 如果距离小于阈值，认为是相邻的
            if distance < distance_threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)
    
    return dict(adjacency)


# ==================== 拼接顺序和方向 ====================

def find_start_region(ocr_regions: List[Dict[str, Any]], 
                     adjacency: Dict[int, List[int]]) -> int:
    """
    找到起始区域（左上角或最大连通分量中的左上角）
    
    Args:
        ocr_regions: OCR识别结果
        adjacency: 邻接图
    
    Returns:
        起始区域的索引
    """
    # 如果没有邻接关系，选择左上角
    if not adjacency:
        # 选择左上角的区域（min(x + y)）
        min_sum = float('inf')
        start_idx = 0
        for idx, region in enumerate(ocr_regions):
            bbox = region.get("bbox", {})
            center = bbox.get("center", [0, 0])
            if isinstance(center, list) and len(center) >= 2:
                sum_xy = center[0] + center[1]
                if sum_xy < min_sum:
                    min_sum = sum_xy
                    start_idx = idx
        return start_idx
    
    # 找到最大连通分量
    visited = set()
    max_component = []
    
    def dfs(node: int, component: List[int]):
        if node in visited:
            return
        visited.add(node)
        component.append(node)
        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, component)
    
    for region_idx in range(len(ocr_regions)):
        if region_idx not in visited:
            component = []
            dfs(region_idx, component)
            if len(component) > len(max_component):
                max_component = component
    
    # 在最大连通分量中选择左上角
    if max_component:
        min_sum = float('inf')
        start_idx = max_component[0]
        for idx in max_component:
            bbox = ocr_regions[idx].get("bbox", {})
            center = bbox.get("center", [0, 0])
            if isinstance(center, list) and len(center) >= 2:
                sum_xy = center[0] + center[1]
                if sum_xy < min_sum:
                    min_sum = sum_xy
                    start_idx = idx
        return start_idx
    
    return 0


def calculate_align(region_a: Dict[str, Any], region_b: Dict[str, Any]) -> str:
    """
    计算两个区域的拼接对齐方式（基于边界框中心，备用方法）
    
    Args:
        region_a: 参考区域
        region_b: 目标区域
    
    Returns:
        对齐方式: "right", "left", "top", "bottom"
    """
    bbox_a = region_a.get("bbox", {})
    bbox_b = region_b.get("bbox", {})
    center_a = bbox_a.get("center", [0, 0])
    center_b = bbox_b.get("center", [0, 0])
    
    if not isinstance(center_a, list) or len(center_a) < 2:
        center_a = [0, 0]
    if not isinstance(center_b, list) or len(center_b) < 2:
        center_b = [0, 0]
    
    dx = center_b[0] - center_a[0]
    dy = center_b[1] - center_a[1]
    
    # 如果水平距离大于垂直距离，判定为水平关系
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "bottom" if dy > 0 else "top"


def calculate_alignment_from_shared_points(
    region_a: Dict[str, Any],
    region_b: Dict[str, Any],
    shared_points: List[Dict[str, Any]]
) -> Tuple[str, List[int], int]:
    """
    基于共享的global_indices点计算两个区域的对齐方式、偏移量和重叠
    
    Args:
        region_a: 参考区域
        region_b: 目标区域
        shared_points: 共享点列表，每个元素包含 {"pixel": (x, y), ...}
    
    Returns:
        (align, offset, overlap): 对齐方式、偏移量[x, y]、重叠像素数
    """
    if not shared_points:
        # 如果没有共享点，回退到基于中心的方法
        align = calculate_align(region_a, region_b)
        return align, [0, 0], 0
    
    bbox_a = region_a.get("bbox", {})
    bbox_b = region_b.get("bbox", {})
    
    min_x_a = bbox_a.get("min_x", 0)
    max_x_a = bbox_a.get("max_x", 0)
    min_y_a = bbox_a.get("min_y", 0)
    max_y_a = bbox_a.get("max_y", 0)
    
    min_x_b = bbox_b.get("min_x", 0)
    max_x_b = bbox_b.get("max_x", 0)
    min_y_b = bbox_b.get("min_y", 0)
    max_y_b = bbox_b.get("max_y", 0)
    
    # 提取所有共享点的坐标
    pixels = [pt["pixel"] for pt in shared_points]
    
    # 计算共享点在两个区域中的相对位置
    # 对于区域A：计算共享点在区域A边界框中的位置
    # 对于区域B：计算共享点在区域B边界框中的位置
    
    # 计算共享点相对于区域A的边界位置
    # 找出共享点最接近区域A的哪条边
    min_dist_to_top_a = min(abs(y - min_y_a) for x, y in pixels)
    min_dist_to_bottom_a = min(abs(y - max_y_a) for x, y in pixels)
    min_dist_to_left_a = min(abs(x - min_x_a) for x, y in pixels)
    min_dist_to_right_a = min(abs(x - max_x_a) for x, y in pixels)
    
    # 计算共享点相对于区域B的边界位置
    min_dist_to_top_b = min(abs(y - min_y_b) for x, y in pixels)
    min_dist_to_bottom_b = min(abs(y - max_y_b) for x, y in pixels)
    min_dist_to_left_b = min(abs(x - min_x_b) for x, y in pixels)
    min_dist_to_right_b = min(abs(x - max_x_b) for x, y in pixels)
    
    # 确定对齐方向：共享点最接近的边界
    # 如果共享点接近区域A的顶部和区域B的底部，说明区域B在区域A的上方，align="top"
    # 如果共享点接近区域A的底部和区域B的顶部，说明区域B在区域A的下方，align="bottom"
    # 如果共享点接近区域A的左侧和区域B的右侧，说明区域B在区域A的左侧，align="left"
    # 如果共享点接近区域A的右侧和区域B的左侧，说明区域B在区域A的右侧，align="right"
    distances = {
        "top": (min_dist_to_top_a + min_dist_to_bottom_b, "top"),      # B在A上方
        "bottom": (min_dist_to_bottom_a + min_dist_to_top_b, "bottom"), # B在A下方
        "left": (min_dist_to_left_a + min_dist_to_right_b, "left"),    # B在A左侧
        "right": (min_dist_to_right_a + min_dist_to_left_b, "right")   # B在A右侧
    }
    
    # 选择距离和最小的对齐方式
    best_align_key = min(distances.keys(), key=lambda k: distances[k][0])
    align = distances[best_align_key][1]  # 区域B相对于区域A的对齐方式
    
    # 计算偏移量
    # 偏移量应该使共享点在拼接后对齐
    # 但是由于UV展开，共享点在两个区域中的位置可能不同
    # 我们计算使得共享点对齐所需的偏移
    offset_x, offset_y = 0, 0
    
    # 由于共享点表示3D空间中的同一位置，在拼接时应该重合
    # 但在UV空间中它们的位置可能不同，这反映了UV展开的特性
    # 因此，我们基于边界框的位置关系来计算偏移
    
    if align == "right":
        # 区域B在区域A的右侧
        # 偏移量应该使得区域B的左边界紧挨着区域A的右边界
        # 考虑共享点，如果有重叠则需要调整
        offset_x = int(max_x_a - min_x_b)
    
    elif align == "left":
        # 区域B在区域A的左侧
        offset_x = int(min_x_a - max_x_b)
    
    elif align == "bottom":
        # 区域B在区域A的下方
        offset_y = int(max_y_a - min_y_b)
    
    elif align == "top":
        # 区域B在区域A的上方
        offset_y = int(min_y_a - max_y_b)
    
    # 计算重叠量
    # 重叠量是两个区域在拼接方向上重叠的像素数
    # 基于共享点的分布来判断是否有实际重叠
    overlap = 0
    
    # 分析共享点相对于边界的分布
    # 如果共享点接近边界，说明两个区域在边界处连接，可能有少量重叠
    # 如果共享点在内部，说明存在重叠
    
    if align in ["right", "left"]:
        # 水平方向
        # 检查共享点是否在重叠区域内
        if align == "right":
            # 重叠区域：max_x_a 到 min_x_b 之间
            overlap_region_start = max_x_a
            overlap_region_end = min_x_b
        else:
            # 重叠区域：max_x_b 到 min_x_a 之间
            overlap_region_start = max_x_b
            overlap_region_end = min_x_a
        
        if overlap_region_end > overlap_region_start:
            # 存在重叠区域
            # 计算有多少共享点在这个重叠区域内
            points_in_overlap = sum(1 for x, y in pixels 
                                   if overlap_region_start <= x <= overlap_region_end)
            
            if points_in_overlap > 0:
                # 有共享点在重叠区域，计算实际重叠量
                overlap = int(overlap_region_end - overlap_region_start)
                # 限制重叠量，避免过大
                overlap = min(overlap, 50)  # 最大重叠50像素
    else:
        # 垂直方向
        if align == "bottom":
            overlap_region_start = max_y_a
            overlap_region_end = min_y_b
        else:
            overlap_region_start = max_y_b
            overlap_region_end = min_y_a
        
        if overlap_region_end > overlap_region_start:
            points_in_overlap = sum(1 for x, y in pixels 
                                   if overlap_region_start <= y <= overlap_region_end)
            
            if points_in_overlap > 0:
                overlap = int(overlap_region_end - overlap_region_start)
                overlap = min(overlap, 50)  # 最大重叠50像素
    
    return align, [offset_x, offset_y], overlap


def bfs_traverse_and_generate_order(start_region: int, 
                                   adjacency: Dict[int, List[int]],
                                   ocr_regions: List[Dict[str, Any]],
                                   shared_points_map: Optional[Dict[Tuple[int, int], List[Dict[str, Any]]]] = None) -> List[Dict[str, Any]]:
    """
    广度优先遍历邻接图，生成拼接顺序
    
    Args:
        start_region: 起始区域索引
        adjacency: 邻接图
        ocr_regions: OCR识别结果
    
    Returns:
        merge_order: 拼接顺序配置列表
    """
    queue = [start_region]
    visited = {start_region}
    merge_order = []
    
    # 第一个区域作为anchor
    merge_order.append({
        "region_index": start_region,
        "name": f"region_{start_region}",
        "transform": {
            "rotation": 0,
            "scale": 1.0,
            "flip_horizontal": False,
            "flip_vertical": True
        },
        "position": {
            "type": "anchor",
            "x": 0,
            "y": 0
        }
    })
    
    # 维护在merge_order中的索引映射
    order_index_map = {start_region: 0}
    
    while queue:
        current = queue.pop(0)
        current_order_idx = order_index_map[current]
        
        # 获取相邻区域，按空间位置排序（优先处理最近的）
        neighbors = adjacency.get(current, [])
        
        # 按与当前区域的距离排序
        def distance_key(neighbor_idx):
            bbox_a = ocr_regions[current].get("bbox", {})
            bbox_b = ocr_regions[neighbor_idx].get("bbox", {})
            center_a = bbox_a.get("center", [0, 0])
            center_b = bbox_b.get("center", [0, 0])
            if isinstance(center_a, list) and isinstance(center_b, list) and len(center_a) >= 2 and len(center_b) >= 2:
                dx = center_b[0] - center_a[0]
                dy = center_b[1] - center_a[1]
                return dx * dx + dy * dy
            return float('inf')
        
        neighbors.sort(key=distance_key)
        
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                
                # 获取共享点（如果可用）
                shared_points = None
                if shared_points_map:
                    # 尝试获取共享点（注意顺序可能不同）
                    key1 = (min(current, neighbor), max(current, neighbor))
                    if key1 not in shared_points_map:
                        # 尝试反向
                        key1 = (max(current, neighbor), min(current, neighbor))
                    if key1 in shared_points_map:
                        shared_points = shared_points_map[key1]
                
                # 基于共享点计算对齐方式、偏移量和重叠
                if shared_points:
                    align, offset, overlap = calculate_alignment_from_shared_points(
                        ocr_regions[current],
                        ocr_regions[neighbor],
                        shared_points
                    )
                else:
                    # 回退到基于中心的方法
                    align = calculate_align(ocr_regions[current], ocr_regions[neighbor])
                    offset = [0, 0]
                    overlap = 0
                
                # 添加到merge_order
                order_idx = len(merge_order)
                merge_order.append({
                    "region_index": neighbor,
                    "name": f"region_{neighbor}",
                    "transform": {
                        "rotation": 0,
                        "scale": 1.0,
                        "flip_horizontal": False,
                        "flip_vertical": True
                    },
                    "position": {
                        "type": "relative",
                        "relative_to": current_order_idx,  # 相对于current在merge_order中的位置
                        "align": align,
                        "offset": offset,
                        "overlap": overlap
                    }
                })
                
                order_index_map[neighbor] = order_idx
                queue.append(neighbor)
    
    # 处理未连接的区域（孤立区域）
    for region_idx in range(len(ocr_regions)):
        if region_idx not in visited:
            # 作为独立区域添加到末尾，智能判断相对于哪个已添加的区域
            if merge_order:
                # 找到最近的已添加区域
                current_region = ocr_regions[region_idx]
                best_ref_idx = 0
                best_ref_order_idx = 0
                min_distance = float('inf')
                
                for ref_order_idx, ref_config in enumerate(merge_order):
                    ref_region_idx = ref_config["region_index"]
                    ref_region = ocr_regions[ref_region_idx]
                    
                    # 计算距离
                    center_a = ref_region.get("bbox", {}).get("center", [0, 0])
                    center_b = current_region.get("bbox", {}).get("center", [0, 0])
                    if isinstance(center_a, list) and isinstance(center_b, list) and len(center_a) >= 2 and len(center_b) >= 2:
                        dx = center_b[0] - center_a[0]
                        dy = center_b[1] - center_a[1]
                        distance = dx * dx + dy * dy
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_ref_idx = ref_region_idx
                            best_ref_order_idx = ref_order_idx
                
                # 尝试获取共享点
                shared_points = None
                if shared_points_map:
                    key1 = (min(best_ref_idx, region_idx), max(best_ref_idx, region_idx))
                    if key1 not in shared_points_map:
                        key1 = (max(best_ref_idx, region_idx), min(best_ref_idx, region_idx))
                    if key1 in shared_points_map:
                        shared_points = shared_points_map[key1]
                
                # 计算对齐方式
                if shared_points:
                    align, offset, overlap = calculate_alignment_from_shared_points(
                        ocr_regions[best_ref_idx],
                        ocr_regions[region_idx],
                        shared_points
                    )
                else:
                    align = calculate_align(ocr_regions[best_ref_idx], ocr_regions[region_idx])
                    offset = [0, 0]
                    overlap = 0
                last_idx = best_ref_order_idx
            else:
                last_idx = -1
                align = None
                offset = [0, 0]
                overlap = 0
            
            merge_order.append({
                "region_index": region_idx,
                "name": f"region_{region_idx}",
                "transform": {
                    "rotation": 0,
                    "scale": 1.0,
                    "flip_horizontal": False,
                    "flip_vertical": True
                },
                "position": {
                    "type": "relative" if last_idx >= 0 else "anchor",
                    "relative_to": last_idx if last_idx >= 0 else None,
                    "align": align if align else None,
                    "offset": offset if last_idx >= 0 else [0, 0],
                    "overlap": overlap if last_idx >= 0 else 0
                } if last_idx >= 0 else {
                    "type": "anchor",
                    "x": 0,
                    "y": 0
                }
            })
    
    return merge_order


# ==================== 可视化锚点 ====================

def uv_to_pixel(uv: Tuple[float, float], texture_size: List[int]) -> Tuple[int, int]:
    """
    将UV坐标转换为像素坐标
    
    Args:
        uv: UV坐标 (u, v)
        texture_size: 纹理尺寸 [width, height]
    
    Returns:
        像素坐标 (x, y)
    """
    u, v = uv
    w, h = texture_size
    x = int(u * w)
    y = int((1.0 - v) * h)  # OpenGL风格，v=0在顶部，需要翻转
    return (x, y)


def find_shared_global_indices_points(
    region_to_global_indices: Dict[int, Set[int]],
    region_to_faces: Dict[int, Set[Tuple[int, int]]],
    uv_info: List[Dict[str, Any]]
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """
    找到区域间共享的global_indices对应的UV坐标点
    
    Args:
        region_to_global_indices: {region_index: {global_index, ...}}
        region_to_faces: {region_index: {(block_idx, face_id), ...}}
        uv_info: UV信息列表
    
    Returns:
        shared_points: {(region_i, region_j): [(block_idx, face_idx, uv_coord, pixel_coord), ...]}
    """
    shared_points = defaultdict(list)
    region_list = list(region_to_global_indices.keys())
    
    # 建立 global_index -> (block_idx, face_idx, vertex_idx, uv) 的映射
    global_index_to_uv = defaultdict(list)
    
    for block_idx, block in enumerate(uv_info):
        texture_size = block.get("texture_size", [1024, 1024])
        faces = block.get("faces", [])
        
        for face_idx, face in enumerate(faces):
            global_indices = face.get("global_indices", [])
            uvs = face.get("uvs", [])
            
            if len(global_indices) == len(uvs):
                for vertex_idx, (gidx, uv) in enumerate(zip(global_indices, uvs)):
                    global_index_to_uv[gidx].append({
                        "block_idx": block_idx,
                        "face_idx": face_idx,
                        "vertex_idx": vertex_idx,
                        "uv": tuple(uv),
                        "texture_size": texture_size,
                        "pixel": uv_to_pixel(tuple(uv), texture_size)
                    })
    
    # 找到每对区域共享的global_indices
    for i, region_i in enumerate(region_list):
        indices_i = region_to_global_indices[region_i]
        
        for j, region_j in enumerate(region_list[i+1:], i+1):
            indices_j = region_to_global_indices[region_j]
            
            # 找到交集
            shared_indices = indices_i & indices_j
            
            if shared_indices:
                # 为每个共享的global_index找到对应的UV坐标
                # 使用集合来去重，避免同一个位置被标记多次
                seen_pixels = set()
                
                for gidx in shared_indices:
                    if gidx in global_index_to_uv:
                        # 对于每个global_index，可能出现在多个block/face中
                        # 我们为每个唯一的位置（像素坐标）添加一个点
                        for uv_info_item in global_index_to_uv[gidx]:
                            pixel = uv_info_item["pixel"]
                            pixel_key = pixel  # (x, y) 作为唯一标识
                            
                            # 如果这个像素位置还没有被标记过，则添加
                            if pixel_key not in seen_pixels:
                                seen_pixels.add(pixel_key)
                                shared_points[(region_i, region_j)].append({
                                    "global_index": gidx,
                                    "block_idx": uv_info_item["block_idx"],
                                    "face_idx": uv_info_item["face_idx"],
                                    "vertex_idx": uv_info_item["vertex_idx"],
                                    "uv": uv_info_item["uv"],
                                    "pixel": pixel
                                })
    
    return dict(shared_points)


def visualize_shared_indices(
    image_file: str,
    pixel_boundry_file: str,
    uv_info_file: str,
    output_image_file: str
) -> None:
    """
    在图像上可视化区域间共享的global_indices对应的点
    
    Args:
        image_file: 原始图像文件路径
        pixel_boundry_file: OCR识别结果JSON文件路径
        uv_info_file: UV信息JSON文件路径
        output_image_file: 输出的可视化图像路径
    """
    print("=" * 60)
    print("🎨 开始可视化共享的global_indices点...")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n📂 加载数据文件...")
    with open(pixel_boundry_file, "r", encoding="utf-8") as f:
        ocr_regions = json.load(f)
    
    with open(uv_info_file, "r", encoding="utf-8") as f:
        uv_info = json.load(f)
    
    # 加载图像
    image = cv2.imread(image_file)
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_file}")
    
    # 复制图像用于绘制
    vis_image = image.copy()
    
    # 2. 建立映射
    print("\n🔍 建立区域到三角面的映射...")
    region_to_faces = find_regions_faces(ocr_regions, uv_info)
    
    print("\n🔗 提取global_indices...")
    region_to_global_indices = get_region_global_indices(region_to_faces, uv_info)
    
    print("\n📍 查找共享的global_indices点...")
    shared_points = find_shared_global_indices_points(
        region_to_global_indices,
        region_to_faces,
        uv_info
    )
    
    # 3. 绘制区域边界框
    colors = [
        (0, 255, 0),    # 绿色 - 区域0
        (255, 0, 0),    # 蓝色 - 区域1
        (0, 0, 255),    # 红色 - 区域2
        (255, 255, 0),  # 青色 - 区域3
        (255, 0, 255),  # 洋红 - 区域4
        (0, 255, 255),  # 黄色 - 区域5
    ]
    
    print("\n🎨 绘制区域边界框...")
    for region_idx, region in enumerate(ocr_regions):
        bbox = region.get("bbox", {})
        color = colors[region_idx % len(colors)]
        
        # 绘制边界框
        min_x = int(bbox.get("min_x", 0))
        min_y = int(bbox.get("min_y", 0))
        max_x = int(bbox.get("max_x", 0))
        max_y = int(bbox.get("max_y", 0))
        
        cv2.rectangle(vis_image, (min_x, min_y), (max_x, max_y), color, 2)
        
        # 添加区域标签
        label = f"Region {region_idx}"
        cv2.putText(vis_image, label, (min_x, min_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 4. 绘制共享点
    print("\n🎨 绘制共享的global_indices点...")
    point_colors = {
        (0, 1): (255, 255, 255),  # 白色 - 区域0和1的共享点
        (0, 2): (255, 255, 255),  # 白色
        (1, 2): (255, 255, 255),  # 白色
    }
    
    total_points = 0
    for (region_i, region_j), points in shared_points.items():
        color = point_colors.get((region_i, region_j), (255, 255, 255))
        
        print(f"   区域 {region_i} 与 区域 {region_j} 共享 {len(points)} 个点")
        
        for point_info in points:
            pixel = point_info["pixel"]
            x, y = pixel
            
            # 绘制点（较大的圆圈）
            cv2.circle(vis_image, (x, y), 5, color, -1)  # 实心圆
            cv2.circle(vis_image, (x, y), 8, color, 2)    # 外圈
            
            total_points += 1
    
    # 5. 添加图例
    legend_y = 30
    cv2.putText(vis_image, "Shared Global Indices Points", (10, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    legend_y += 25
    
    for (region_i, region_j), points in shared_points.items():
        text = f"Region {region_i} <-> Region {region_j}: {len(points)} points"
        cv2.putText(vis_image, text, (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 20
    
    # 6. 保存图像
    output_dir = os.path.dirname(output_image_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(output_image_file, vis_image)
    
    print(f"\n✅ 可视化完成！")
    print(f"   总共找到 {len(shared_points)} 对区域有共享点")
    print(f"   总共绘制了 {total_points} 个共享点")
    print(f"   输出图像: {output_image_file}")
    print("=" * 60)


# ==================== 主函数 ====================

def auto_generate_merge_config(
    pixel_boundry_file: str,
    uv_info_file: str,
    output_config_file: str
) -> None:
    """
    自动生成拼接配置文件
    
    Args:
        pixel_boundry_file: OCR识别结果JSON文件路径
        uv_info_file: UV信息JSON文件路径
        output_config_file: 输出的配置文件名路径
    """
    print("=" * 60)
    print("🚀 开始自动生成拼接配置...")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n📂 步骤1: 加载数据文件...")
    with open(pixel_boundry_file, "r", encoding="utf-8") as f:
        ocr_regions = json.load(f)
    
    with open(uv_info_file, "r", encoding="utf-8") as f:
        uv_info = json.load(f)
    
    print(f"   ✅ 加载了 {len(ocr_regions)} 个OCR区域")
    print(f"   ✅ 加载了 {len(uv_info)} 个UV blocks")
    
    # 2. 为每个区域找到对应的三角面和global_indices
    print("\n🔍 步骤2: 建立区域到三角面的映射...")
    region_to_faces = find_regions_faces(ocr_regions, uv_info)
    print(f"   ✅ 完成区域-三角面映射")
    
    found_count = 0
    for region_idx in range(len(ocr_regions)):
        faces = region_to_faces.get(region_idx, set())
        if faces:
            print(f"   区域 {region_idx}: {len(faces)} 个三角面")
            found_count += 1
        else:
            print(f"   ⚠️  区域 {region_idx}: 未找到对应的三角面")
    
    if found_count == 0:
        print("   ⚠️  警告：没有找到任何区域与三角面的映射，将使用备用策略")
    
    # 3. 获取每个区域的global_indices
    print("\n🔗 步骤3: 提取global_indices...")
    region_to_global_indices = get_region_global_indices(region_to_faces, uv_info)
    
    indices_found = False
    for region_idx in range(len(ocr_regions)):
        indices = region_to_global_indices.get(region_idx, set())
        if indices:
            print(f"   区域 {region_idx}: {len(indices)} 个global_indices")
            indices_found = True
    
    if not indices_found:
        print("   ⚠️  警告：没有找到任何global_indices！")
    
    # 4. 构建邻接图
    print("\n🌐 步骤4: 构建区域邻接图...")
    adjacency = build_adjacency_graph(region_to_global_indices)
    
    print(f"   ✅ 通过global_index构建完成，发现 {len(adjacency)} 个区域有邻接关系")
    if adjacency:
        for region_idx, neighbors in adjacency.items():
            print(f"   区域 {region_idx} 相邻: {neighbors}")
    else:
        print("   ⚠️  未找到global_index邻接关系，使用基于像素位置的备用策略")
        # 使用基于位置的备用策略
        adjacency = build_fallback_adjacency_by_position(ocr_regions, distance_threshold=200.0)
        print(f"   ✅ 基于像素位置构建完成，发现 {len(adjacency)} 个区域有邻接关系")
        if adjacency:
            for region_idx, neighbors in adjacency.items():
                print(f"   区域 {region_idx} 相邻（基于位置）: {neighbors}")
        else:
            print("   ⚠️  未找到邻接关系，将按像素位置顺序拼接")
    
    # 5. 确定起点
    print("\n📍 步骤5: 确定起始区域...")
    # start_region = find_start_region(ocr_regions, adjacency)
    start_region = 2
    print(f"   ✅ 起始区域: {start_region}")
    
    # 6. 获取共享点映射
    print("\n📍 步骤6: 分析共享的global_indices点...")
    shared_points_map = find_shared_global_indices_points(
        region_to_global_indices,
        region_to_faces,
        uv_info
    )
    
    if shared_points_map:
        print(f"   ✅ 找到 {len(shared_points_map)} 对区域有共享点")
        for (ri, rj), points in shared_points_map.items():
            print(f"      区域 {ri} <-> 区域 {rj}: {len(points)} 个共享点")
    else:
        print("   ⚠️  未找到共享点，将使用基于边界框的方法")
    
    # 7. 生成拼接顺序
    print("\n🔄 步骤7: 生成拼接顺序...")
    merge_order = bfs_traverse_and_generate_order(start_region, adjacency, ocr_regions, shared_points_map)
    print(f"   ✅ 生成 {len(merge_order)} 个区域的拼接配置")
    
    # 8. 生成配置文件
    print("\n💾 步骤8: 保存配置文件...")
    config = {
        "_comment": {
            "说明": "自动生成的区域拼接配置文件",
            "生成方式": "基于OCR区域、UV信息和global_index邻接关系自动生成"
        },
        "merge_order": merge_order,
        "output": {
            "size": "auto",
            "background_color": [255, 255, 255]
        }
    }
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_config_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print(f"   ✅ 配置文件已保存到: {output_config_file}")
    print("\n" + "=" * 60)
    print("✅ 自动生成配置完成！")
    print("=" * 60)


# ==================== 主程序 ====================

if __name__ == "__main__":
    # 配置路径
    pixel_boundry_file = r"D:\myProject\NAN\output\pixel_boundry.json"
    uv_info_file = r"D:\myProject\NAN\data\Tile_+028_+014_L22_0005000_uv_info.json"
    output_config_file = r"D:\myProject\NAN\output\merge_config_auto.json"
    image_file = r"D:\myProject\NAN\data\Tile_+028_+014_L22_0005000.jpg"
    output_vis_image = r"D:\myProject\NAN\output\shared_indices_visualization.jpg"
    
    # 生成拼接配置
    auto_generate_merge_config(
        pixel_boundry_file,
        uv_info_file,
        output_config_file
    )
    
    # 可视化共享的global_indices点
    print("\n" + "=" * 60)
    visualize_shared_indices(
        image_file,
        pixel_boundry_file,
        uv_info_file,
        output_vis_image
    )