#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成肉眼检查用的预览图: 假彩色(NIR-Red合成) / NDVI / 植被掩膜 三联图"""
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with rasterio.open('input.tif') as s:
    red = s.read(1).astype('float64')
    nir = s.read(2).astype('float64')
with rasterio.open('output_ndvi.tif') as s:
    ndvi = s.read(1)
with rasterio.open('output_vegetation_mask.tif') as s:
    mask = s.read(1)

# 假彩色: NIR->红 Red->绿, 植被在此呈红色(遥感惯例)
def stretch(a):
    v = a[a > 0]
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    return np.clip((a - lo) / (hi - lo), 0, 1)

fc = np.dstack([stretch(nir), stretch(red), np.zeros_like(red)])

ndvi_show = np.ma.masked_equal(ndvi, -9999.0)
mask_show = np.ma.masked_equal(mask, 255)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(fc); ax[0].set_title('False Color (NIR=R, Red=G)\nvegetation = red')
im1 = ax[1].imshow(ndvi_show, cmap='RdYlGn', vmin=-1, vmax=1)
ax[1].set_title('NDVI'); fig.colorbar(im1, ax=ax[1], fraction=0.046)
ax[2].imshow(mask_show, cmap='Greens', vmin=0, vmax=1)
ax[2].set_title('Vegetation Mask (green=veg)')
for a in ax: a.axis('off')
plt.tight_layout()
plt.savefig('preview.png', dpi=100, bbox_inches='tight')
print('已保存 preview.png')

# 交叉验证: 假彩色里偏红(NIR>>Red)的地方应被判为植被
veg = mask == 1
ratio_veg = (nir[veg] / (red[veg] + 1e-9)).mean()
ratio_nonveg = (nir[mask == 0] / (red[mask == 0] + 1e-9)).mean()
print(f'植被像素 NIR/Red 均值={ratio_veg:.2f}, 非植被={ratio_nonveg:.2f}')
print('植被区 NIR/Red 应明显>1 且高于非植被区 ->', 'OK' if ratio_veg > ratio_nonveg > 0 else 'CHECK')
