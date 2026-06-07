#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
卫星影像植被分析工具 (NDVI)

用法:
    python ndvi_tool.py input.tif [--red 1] [--nir 2] [--threshold 0.2]
                        [--outdir .] [--scale S] [--offset O]

设计要点（与 spec.md 对应）:
  1. uint16 → float64 计算，避免整型下溢。
  2. 先把 DN 转反射率 (reflectance = DN*SCALE + OFFSET) 再算 NDVI；
     因为本影像 OFFSET=-0.2 != 0，用原始 DN 直算会系统性错误。
  3. nodata 像素(DN=0) 和 分母为0 的像素 → 输出 -9999，绝不产生假植被或 NaN。
  4. 分块(block_windows)读写，使工具可扩展到大于内存的影像。
  5. 输出保留输入的 CRS 与 transform，地理参考不丢失。
"""

import argparse
import sys
import numpy as np
import rasterio

# NDVI 输出的 nodata 值。不能用 0，因为 0 是合法 NDVI。
NDVI_NODATA = -9999.0
# 植被掩膜中无效像素的标记值（uint8，区别于 0/1）。
MASK_NODATA = 255


def compute_ndvi_block(red_dn, nir_dn, src_nodata, scale, offset):
    """对一个数据块计算 NDVI。

    参数均为某个 window 读出的原始 DN 数组(任意整型/浮点)。
    返回 float32 的 NDVI 块，无效像素填 NDVI_NODATA。

    步骤:
      a) 转 float64：避免 uint16 减法下溢，且保证除法精度。
      b) 找出有效像素：任一波段为 nodata(DN=0) 的像素视为无效。
      c) DN → 反射率：reflectance = DN*scale + offset。
      d) NDVI = (NIR-Red)/(NIR+Red)，仅在分母!=0 且像素有效处计算。
      e) 其余位置填 NDVI_NODATA。
    """
    red = red_dn.astype('float64')
    nir = nir_dn.astype('float64')

    # (b) 有效掩膜：两个波段都不是填充值。src_nodata 可能为 None。
    if src_nodata is not None:
        valid = (red != src_nodata) & (nir != src_nodata)
    else:
        valid = np.ones(red.shape, dtype=bool)

    # (c) 转反射率。OFFSET!=0 时这一步不可省略。
    red_rho = red * scale + offset
    nir_rho = nir * scale + offset

    # (d) 分母，防除零。
    denom = nir_rho + red_rho
    valid = valid & (denom != 0)

    # 先全部填 nodata，再只在有效处写入计算值，天然规避 NaN/Inf。
    ndvi = np.full(red.shape, NDVI_NODATA, dtype='float64')
    np.divide(nir_rho - red_rho, denom, out=ndvi, where=valid)

    return ndvi.astype('float32'), valid


def main():
    ap = argparse.ArgumentParser(description="卫星影像 NDVI 植被分析工具")
    ap.add_argument('input', help='输入 GeoTIFF 路径')
    ap.add_argument('--red', type=int, default=1, help='红光波段索引(1-based)，默认1')
    ap.add_argument('--nir', type=int, default=2, help='近红外波段索引(1-based)，默认2')
    ap.add_argument('--threshold', type=float, default=0.2, help='植被掩膜 NDVI 阈值，默认0.2')
    ap.add_argument('--outdir', default='.', help='输出目录，默认当前目录')
    ap.add_argument('--scale', type=float, default=None,
                    help='DN→反射率 缩放系数，默认读影像标签 SCALE')
    ap.add_argument('--offset', type=float, default=None,
                    help='DN→反射率 偏移量，默认读影像标签 OFFSET')
    args = ap.parse_args()

    with rasterio.open(args.input) as src:
        # 缩放系数：优先命令行，否则读影像全局标签，否则退化为 1/0（不缩放）。
        tags = src.tags()
        scale = args.scale if args.scale is not None else float(tags.get('SCALE', 1.0))
        offset = args.offset if args.offset is not None else float(tags.get('OFFSET', 0.0))
        src_nodata = src.nodata

        print(f"输入: {args.input}  尺寸 {src.width}x{src.height}  波段数 {src.count}")
        print(f"Red=波段{args.red}({src.descriptions[args.red-1]}), "
              f"NIR=波段{args.nir}({src.descriptions[args.nir-1]})")
        print(f"SCALE={scale}, OFFSET={offset}, nodata={src_nodata}")
        if offset != 0:
            print("  注意: OFFSET!=0，已启用 DN→反射率 转换后再算 NDVI。")

        # 输出 profile：单波段 float32，nodata=-9999，继承 CRS/transform。
        ndvi_profile = src.profile.copy()
        ndvi_profile.update(count=1, dtype='float32', nodata=NDVI_NODATA)
        mask_profile = src.profile.copy()
        mask_profile.update(count=1, dtype='uint8', nodata=MASK_NODATA)

        import os
        os.makedirs(args.outdir, exist_ok=True)
        ndvi_path = os.path.join(args.outdir, 'output_ndvi.tif')
        mask_path = os.path.join(args.outdir, 'output_vegetation_mask.tif')
        stats_path = os.path.join(args.outdir, 'statistics.txt')

        # 统计累加器（流式，跨块累加，不需把整图留在内存）。
        n_total = src.width * src.height
        n_valid = 0
        n_veg = 0
        s_sum = 0.0
        s_sqsum = 0.0
        s_min = np.inf
        s_max = -np.inf
        # NDVI 分级直方图边界
        bins = [-1.0, 0.0, 0.2, 0.4, 0.6, 1.0001]
        hist = np.zeros(len(bins) - 1, dtype='int64')

        with rasterio.open(ndvi_path, 'w', **ndvi_profile) as dst_ndvi, \
             rasterio.open(mask_path, 'w', **mask_profile) as dst_mask:

            # 按输入的内部分块结构迭代；若无分块结构，block_windows 返回整图一块。
            for _, window in src.block_windows(1):
                red_dn = src.read(args.red, window=window)
                nir_dn = src.read(args.nir, window=window)

                ndvi, valid = compute_ndvi_block(red_dn, nir_dn, src_nodata, scale, offset)

                # 植被掩膜
                mask = np.full(ndvi.shape, MASK_NODATA, dtype='uint8')
                mask[valid] = 0
                mask[valid & (ndvi > args.threshold)] = 1

                dst_ndvi.write(ndvi, 1, window=window)
                dst_mask.write(mask, 1, window=window)

                # 流式统计（只统计有效像素）
                v = ndvi[valid]
                if v.size:
                    n_valid += v.size
                    n_veg += int((v > args.threshold).sum())
                    s_sum += float(v.sum())
                    s_sqsum += float((v.astype('float64') ** 2).sum())
                    s_min = min(s_min, float(v.min()))
                    s_max = max(s_max, float(v.max()))
                    hist += np.histogram(v, bins=bins)[0]

        # 汇总统计
        mean = s_sum / n_valid if n_valid else float('nan')
        var = (s_sqsum / n_valid - mean ** 2) if n_valid else float('nan')
        std = var ** 0.5 if var >= 0 else float('nan')
        px_area_m2 = abs(src.transform.a * src.transform.e)  # 单像元面积(m^2)

        lines = []
        lines.append("=== NDVI 统计报告 ===")
        lines.append(f"输入影像: {args.input}")
        lines.append(f"总像素: {n_total}")
        lines.append(f"有效像素: {n_valid}  填充/无效像素: {n_total - n_valid}")
        lines.append(f"植被像素(NDVI>{args.threshold}): {n_veg}  "
                     f"占有效像素 {100.0*n_veg/n_valid:.2f}%" if n_valid else "无有效像素")
        lines.append(f"NDVI  min={s_min:.4f}  max={s_max:.4f}  mean={mean:.4f}  std={std:.4f}")
        lines.append(f"单像元面积: {px_area_m2:.1f} m^2")
        lines.append("NDVI 分级面积分布:")
        labels = ["[-1,0) 水体/非植被", "[0,0.2) 裸地/稀疏", "[0.2,0.4) 中低植被",
                  "[0.4,0.6) 中高植被", "[0.6,1] 高密植被"]
        for lab, cnt in zip(labels, hist):
            lines.append(f"  {lab}: {cnt} 像素  {cnt*px_area_m2/1e4:.3f} 公顷")
        report = "\n".join(lines)

        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(report + "\n")

        print("\n" + report)
        print(f"\n输出已写出:\n  {ndvi_path}\n  {mask_path}\n  {stats_path}")


if __name__ == '__main__':
    main()
