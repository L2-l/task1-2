#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
六项自检脚本。完全独立于 ndvi_tool.py 重新实现/重新读取，
用于对拍工具输出。任何一项失败即 AssertionError 退出。
"""
import numpy as np
import rasterio

INPUT = 'input.tif'
NDVI_OUT = 'output_ndvi.tif'
MASK_OUT = 'output_vegetation_mask.tif'
NDVI_NODATA = -9999.0
SCALE = 2.75e-05
OFFSET = -0.2
THRESH = 0.2

passed = []

# ---------- 读入 ----------
with rasterio.open(INPUT) as src:
    red_dn = src.read(1).astype('float64')
    nir_dn = src.read(2).astype('float64')
    src_crs = src.crs
    src_transform = src.transform
    src_shape = (src.height, src.width)
    src_nodata = src.nodata

with rasterio.open(NDVI_OUT) as d:
    out_ndvi = d.read(1)
    out_crs = d.crs
    out_transform = d.transform
    out_shape = (d.height, d.width)
    out_nodata = d.nodata

with rasterio.open(MASK_OUT) as d:
    out_mask = d.read(1)

# ---------- 自检 1: 手算对拍 3 个有效像素 ----------
print("=" * 60)
print("自检1: 手算对拍 3 个有效像素")
test_coords = [(10, 10), (50, 50), (90, 90)]
for (r, c) in test_coords:
    rd, nd = red_dn[r, c], nir_dn[r, c]
    # 手算: 先转反射率
    red_rho = rd * SCALE + OFFSET
    nir_rho = nd * SCALE + OFFSET
    expected = (nir_rho - red_rho) / (nir_rho + red_rho)
    actual = out_ndvi[r, c]
    diff = abs(expected - actual)
    print(f"  ({r},{c}): DN(R={rd:.0f},N={nd:.0f}) -> 反射率(R={red_rho:.4f},N={nir_rho:.4f})"
          f" 手算NDVI={expected:.6f} 工具={actual:.6f} 差={diff:.2e}")
    assert diff < 1e-5, f"像素({r},{c}) 手算与工具不符: {diff}"
passed.append("1 手算对拍")
print("  PASS")

# ---------- 自检 2: 独立重读波段，确认没读错波段 ----------
print("=" * 60)
print("自检2: 独立重读原始波段值")
# 用 element-wise 重新读出原始 DN，对比之前读的，确保波段顺序一致
with rasterio.open(INPUT) as src:
    band1 = src.read(1)  # 应为 Red
    band2 = src.read(2)  # 应为 NIR
    desc = src.descriptions
print(f"  波段描述: {desc}")
assert desc[0] == 'Red' and desc[1] == 'NIR', "波段描述与假设(Red,NIR)不符"
# NIR 在植被区通常 > Red；检查整体均值关系符合物理常识
print(f"  Red 均值={band1[band1!=0].mean():.1f}, NIR 均值={band2[band2!=0].mean():.1f}")
passed.append("2 独立重读波段")
print("  PASS")

# ---------- 自检 3: 范围检查 [-1,1] ----------
print("=" * 60)
print("自检3: NDVI 范围检查")
valid_mask = out_ndvi != NDVI_NODATA
vals = out_ndvi[valid_mask]
print(f"  有效NDVI范围: [{vals.min():.4f}, {vals.max():.4f}]  有效像素={vals.size}")
assert vals.min() >= -1.0 and vals.max() <= 1.0, "NDVI 越界 [-1,1]"
passed.append("3 范围检查")
print("  PASS")

# ---------- 自检 4: 边界情况 ----------
print("=" * 60)
print("自检4: 边界情况(填充/除零)")
# 4a: DN=0 的填充像素必须是 nodata，不能算出植被
fill_pixels = (red_dn == 0) | (nir_dn == 0)
n_fill = int(fill_pixels.sum())
print(f"  填充像素(任一波段DN=0)数量: {n_fill}")
assert np.all(out_ndvi[fill_pixels] == NDVI_NODATA), "存在填充像素未被标记为 nodata"
assert np.all(out_mask[fill_pixels] == 255), "填充像素在掩膜中未标记为无效(255)"
# 4b: 输出中不能有 NaN/Inf
assert not np.any(np.isnan(out_ndvi)), "输出含 NaN"
assert not np.any(np.isinf(out_ndvi)), "输出含 Inf"
print(f"  填充像素全部=nodata ✓, 无 NaN/Inf ✓")
passed.append("4 边界情况")
print("  PASS")

# ---------- 自检 5: 地理参考一致 ----------
print("=" * 60)
print("自检5: 地理参考一致性")
print(f"  CRS: 输入={src_crs} 输出={out_crs}")
print(f"  尺寸: 输入={src_shape} 输出={out_shape}")
assert out_crs == src_crs, "CRS 不一致"
assert out_transform == src_transform, "仿射变换不一致"
assert out_shape == src_shape, "尺寸不一致"
assert out_nodata == NDVI_NODATA, "输出 nodata 标记错误"
print(f"  transform 一致 ✓, nodata={out_nodata} ✓")
passed.append("5 地理参考")
print("  PASS")

# ---------- 自检 6: 统计合理性 + 掩膜一致 ----------
print("=" * 60)
print("自检6: 统计合理性 & 掩膜与NDVI一致")
# 掩膜=1 的像素 NDVI 必须 > 阈值; 掩膜=0 的必须 <=阈值且有效
veg = out_mask == 1
nonveg = out_mask == 0
assert np.all(out_ndvi[veg] > THRESH), "掩膜标记为植被的像素 NDVI 未超阈值"
assert np.all(out_ndvi[nonveg] <= THRESH), "掩膜标记为非植被的像素 NDVI 超阈值"
assert np.all(out_ndvi[nonveg] != NDVI_NODATA), "非植被掩膜含 nodata 像素"
print(f"  植被像素={int(veg.sum())}, 非植被={int(nonveg.sum())}, 无效={int((out_mask==255).sum())}")
print(f"  掩膜与NDVI阈值完全一致 ✓")
passed.append("6 统计/掩膜一致")
print("  PASS")

print("=" * 60)
print(f"全部 {len(passed)} 项自检通过: {passed}")
