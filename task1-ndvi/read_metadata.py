#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""读取 input.tif 的完整元数据"""

import rasterio
import numpy as np

input_file = 'input.tif'

print("="*60)
print("卫星影像元数据分析")
print("="*60)

with rasterio.open(input_file) as src:
    print(f"\n【基本信息】")
    print(f"  驱动格式: {src.driver}")
    print(f"  影像尺寸: {src.width} x {src.height} 像素")
    print(f"  波段数量: {src.count}")
    print(f"  数据类型: {src.dtypes[0]}")

    print(f"\n【地理参考】")
    print(f"  坐标系 (CRS): {src.crs}")
    print(f"  仿射变换:\n{src.transform}")
    print(f"  边界范围: {src.bounds}")

    print(f"\n【NoData & 缩放】")
    print(f"  NoData 值: {src.nodata}")
    print(f"  缩放系数: {src.scales}")
    print(f"  偏移量: {src.offsets}")

    print(f"\n【波段详情】")
    for i in range(1, src.count + 1):
        print(f"\n  --- 波段 {i} ---")
        print(f"    描述: {src.descriptions[i-1]}")
        print(f"    标签: {src.tags(i)}")
        band_data = src.read(i)
        print(f"    值域: [{band_data.min()}, {band_data.max()}]")
        print(f"    均值: {band_data.mean():.2f}")
        print(f"    数据形状: {band_data.shape}")

    print(f"\n【全局标签】")
    tags = src.tags()
    if tags:
        for key, value in tags.items():
            print(f"  {key}: {value}")
    else:
        print("  (无全局标签)")

    print(f"\n【采样像素验证】")
    print("  选取 3 个像素读取所有波段值：")
    coords = [(10, 10), (50, 50), (90, 90)]
    for idx, (r, c) in enumerate(coords, 1):
        print(f"\n  像素 {idx} - 坐标 (row={r}, col={c}):")
        for band_idx in range(1, src.count + 1):
            val = src.read(band_idx)[r, c]
            print(f"    波段 {band_idx}: {val}")

print("\n" + "="*60)
print("元数据读取完成")
print("="*60)
