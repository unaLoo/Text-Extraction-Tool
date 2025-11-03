# 图像区域提取与拼接工具

基于 OCR 的文本区域识别、提取和智能拼接工具，支持基于 3D 几何信息的自动拼接配置生成。





## 功能特性

- **文本区域识别**：使用 PaddleOCR 自动识别图像中的文本区域
- **区域提取**：根据像素坐标从原图中精确提取指定区域
- **智能拼接**：支持旋转、缩放、翻转等变换，灵活配置拼接布局
- **自动配置生成**：基于 UV 信息和 global_indices 自动生成拼接配置文件
- **几何感知拼接**：利用 3D 模型的顶点映射关系，准确识别区域邻接关系
- **可视化工具**：可视化区域间共享的几何连接点

## 项目结构

```
NAN/
├── extractor.py              # 文本区域提取工具
├── merger.py                 # 区域拼接工具
├── auto_merge_config.py      # 自动生成拼接配置工具
├── merge_config.json         # 拼接配置文件（手动或自动生成）
├── data/                     # 输入数据目录
│   ├── *.jpg                 # 源图像
│   ├── *_uv_info.json        # UV 信息文件
│   └── *_uv_to_global_index.json  # UV 到 global_index 映射
└── output/                   # 输出结果目录
    ├── pixel_boundry.json              # 识别的文本区域坐标
    ├── merge_config_auto.json         # 自动生成的拼接配置
    ├── shared_indices_visualization.jpg # 共享点可视化图像
    └── extracted_regions/             # 提取的区域图像
```

## 环境要求

- Python 3.8+
- OpenCV (`cv2`)
- PaddleOCR
- NumPy

## 快速开始

### 1. 识别文本区域并生成坐标文件

运行 `extractor.py` 识别图像中的文本区域：

```python
python extractor.py
```

**输出**：`output/pixel_boundry.json` - 包含文本区域的坐标、文本内容、置信度等信息

**JSON 格式示例**：
```json
[
    {
        "index": 0,
        "points": [[290.0, 430.0], [420.0, 430.0], [420.0, 468.0], [290.0, 468.0]],
        "text": "识别的文本",
        "confidence": 0.95,
        "bbox": {
            "min_x": 290.0,
            "min_y": 430.0,
            "max_x": 420.0,
            "max_y": 468.0,
            "center": [355.0, 449.0],
            "width": 130.0,
            "height": 38.0
        }
    }
]
```

### 2. 自动生成拼接配置（推荐）

运行 `auto_merge_config.py` 基于 UV 信息和 global_indices 自动生成拼接配置：

```python
python auto_merge_config.py
```

**功能**：
1. 分析 OCR 区域与 UV 三角面的映射关系
2. 通过 shared global_indices 构建区域邻接图
3. 基于共享点计算准确的对齐方式、偏移量和重叠
4. 自动生成 `merge_config_auto.json` 配置文件
5. 生成共享点可视化图像

**输出**：
- `output/merge_config_auto.json` - 自动生成的拼接配置
- `output/shared_indices_visualization.jpg` - 共享点可视化图像

### 3. 提取和拼接区域

运行 `merger.py` 提取区域并按配置拼接：

```python
python merger.py
```

**流程**：
1. 从 `pixel_boundry.json` 读取区域坐标
2. 从原图中提取各个区域
3. 根据 `merge_config.json` 或 `merge_config_auto.json` 配置进行拼接
4. 输出拼接后的图像

## 配置说明

### 自动生成配置 vs 手动配置

- **自动生成（推荐）**：使用 `auto_merge_config.py`，基于 3D 几何信息自动计算拼接关系，更准确
- **手动配置**：直接编辑 `merge_config.json`，适合特殊需求或调试

### 拼接配置文件 (`merge_config.json` / `merge_config_auto.json`)

```json
{
    "merge_order": [
        {
            "region_index": 0,              // 区域索引
            "name": "region_0",              // 区域名称
            "transform": {                   // 变换参数
                "rotation": 0,               // 旋转角度（度）
                "scale": 1.0,               // 缩放比例
                "flip_horizontal": false,   // 水平翻转
                "flip_vertical": false      // 垂直翻转
            },
            "position": {                    // 位置配置
                "type": "anchor",            // "anchor" 或 "relative"
                "x": 0,                      // 锚点坐标（仅 anchor 类型）
                "y": 0,
                "relative_to": 0,            // 参考区域索引（仅 relative 类型）
                "align": "right",            // 对齐方式：right/left/top/bottom/center
                "offset": [0, 0],            // 偏移量 [x, y]
                "overlap": 0                 // 重叠像素数
            }
        }
    ],
    "output": {
        "size": "auto",                      // 画布尺寸："auto" 或 [width, height]
        "background_color": [255, 255, 255]  // 背景颜色 [R, G, B]
    }
}
```

### 位置对齐方式

- `right`：放在参考区域的右侧
- `left`：放在参考区域的左侧
- `top`：放在参考区域的上方
- `bottom`：放在参考区域的下方
- `center`：与参考区域中心对齐

## 使用示例

### 水平拼接示例

```json
{
    "merge_order": [
        {"region_index": 0, "position": {"type": "anchor"}},
        {"region_index": 1, "position": {"type": "relative", "relative_to": 0, "align": "right"}}
    ]
}
```

### 旋转后拼接示例

```json
{
    "merge_order": [
        {"region_index": 0, "position": {"type": "anchor"}},
        {
            "region_index": 1,
            "transform": {"rotation": 90},
            "position": {"type": "relative", "relative_to": 0, "align": "bottom"}
        }
    ]
}
```

## 参数配置

### extractor.py

在文件顶部修改配置：

```python
image_path = r"path/to/your/image.jpg"  # 输入图像路径
output_dir = r"output"                  # 输出目录
```

### auto_merge_config.py

在 `if __name__ == "__main__":` 部分修改：

```python
pixel_boundry_file = r"output/pixel_boundry.json"                    # OCR 识别结果
uv_info_file = r"data/*_uv_info.json"                                # UV 信息文件
output_config_file = r"output/merge_config_auto.json"                # 输出配置路径
image_file = r"data/source_image.jpg"                                 # 源图像（用于可视化）
output_vis_image = r"output/shared_indices_visualization.jpg"        # 可视化图像路径
```

### merger.py

在 `if __name__ == "__main__":` 部分修改：

```python
texture_fp = r"path/to/source/image.jpg"           # 源图像路径
px_boundry_fp = r"output/pixel_boundry.json"       # 坐标文件路径
merge_config_fp = r"output/merge_config_auto.json" # 拼接配置路径（自动生成或手动）
```

## 输出文件

- `output/pixel_boundry.json`：文本区域坐标和识别信息
- `output/merge_config_auto.json`：自动生成的拼接配置文件
- `output/shared_indices_visualization.jpg`：共享 global_indices 点可视化图像
- `output/extracted_regions/region_*.jpg`：提取的各个区域图像
- `output/extracted_regions/$$$$merged_result.jpg`：拼接后的最终图像

## 工作原理

### 自动拼接配置生成流程

1. **区域-三角面映射**：将 OCR 识别的区域边界框映射到 UV 空间的三角面
2. **Global Indices 提取**：从三角面中提取 global_indices（3D 模型的顶点索引）
3. **邻接关系构建**：通过 shared global_indices 识别区域间的几何连接关系
4. **共享点分析**：计算区域间共享的顶点在 UV 空间的位置
5. **拼接参数计算**：基于共享点分布计算对齐方式、偏移量和重叠

### 优势

- **几何准确性**：基于 3D 模型的顶点连接关系，而非简单的像素距离
- **处理复杂布局**：能正确识别 UV 展开后看似不相邻但在 3D 中实际相邻的区域
- **自动优化**：自动计算最优的对齐方式和重叠，减少手动调试

## 注意事项

1. 首次运行需要下载 PaddleOCR 模型，请确保网络连接正常
2. `merge_config.json` 中的 `region_index` 需要与 `pixel_boundry.json` 中的索引对应
3. 相对定位的 `relative_to` 索引是基于 `merge_order` 中的顺序（从 0 开始）
4. 拼接时会自动处理超出边界的区域
5. 自动生成的配置基于 UV 信息和 global_indices，确保数据文件完整
6. 如果无法通过 global_indices 建立邻接关系，会自动回退到基于像素位置的备用策略

## 许可证

本项目仅供学习交流使用。

