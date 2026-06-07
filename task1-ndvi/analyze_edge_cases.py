#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证 DN vs 反射率 计算 NDVI 的差异，并探查边界情况"""
import rasterio
import numpy as np

SCALE = 2.75e-05
OFFSET = -0.2

with rasterio.open('input.tif') as src:
    red_dn = src.read(1).astype('float64')
    nir_dn = src.read(2).astype('float64')

valid = (red_dn != 0) & (nir_dn != 0)
print(f"总像素: {red_dn.size}, 有效像素(DN!=0): {valid.sum()}, 填充像素: {(~valid).sum()}")

# 反射率
red_rho = red_dn * SCALE + OFFSET
nir_rho = nir_dn * SCALE + OFFSET

print(f"\n反射率范围(有效像素):")
print(f"  Red : [{red_rho[valid].min():.4f}, {red_rho[valid].max():.4f}]")
print(f"  NIR : [{nir_rho[valid].min():.4f}, {nir_rho[valid].max():.4f}]")
print(f"  有负反射率的像素? Red<0: {(red_rho[valid]<0).sum()}, NIR<0: {(nir_rho[valid]<0).sum()}")

# 两种 NDVI
ndvi_dn = np.where(valid, (nir_dn - red_dn) / (nir_dn + red_dn + 1e-12), np.nan)
denom_rho = nir_rho + red_rho
ndvi_rho = np.where(valid & (denom_rho != 0), (nir_rho - red_rho) / denom_rho, np.nan)

print(f"\n分母(反射率和)范围: [{denom_rho[valid].min():.4f}, {denom_rho[valid].max():.4f}]")
print(f"  分母<=0 的有效像素数: {(valid & (denom_rho <= 0)).sum()}  <- 若>0 则反射率NDVI存在除零/出界风险")

print(f"\nNDVI(原始DN) 范围: [{np.nanmin(ndvi_dn):.4f}, {np.nanmax(ndvi_dn):.4f}]")
print(f"NDVI(反射率)  范围: [{np.nanmin(ndvi_rho):.4f}, {np.nanmax(ndvi_rho):.4f}]")

# 逐像素对比 3 个采样点
print(f"\n三个采样像素 DN-NDVI vs 反射率-NDVI 对比:")
for (r, c) in [(10, 10), (50, 50), (90, 90)]:
    print(f"  ({r},{c}): DN_red={red_dn[r,c]:.0f} DN_nir={nir_dn[r,c]:.0f} | "
          f"NDVI_DN={ndvi_dn[r,c]:.4f}  NDVI_反射率={ndvi_rho[r,c]:.4f}  "
          f"差异={abs(ndvi_dn[r,c]-ndvi_rho[r,c]):.4f}")
