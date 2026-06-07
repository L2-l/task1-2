"""
make_synthetic_data.py
----------------------
生成一景"有空间连续性"的 4 波段合成影像 + 地面真值标签,用于在拿到真数据前
跑通整条流程,并刻意复现"随机切分 vs 空间切分"那一课。

关键点:地物必须成片(spatially contiguous),否则空间切分和随机切分就没区别,
也就演示不出数据泄漏。这里用低频噪声平滑出大块连续区域来模拟真实地物分布。

波段约定(和 SPEC.md 一致):
  Band 1 = Red, Band 2 = NIR, Band 3 = Green, Band 4 = SWIR
标签约定: 0 = 其他, 1 = 植被, 2 = 水体
"""

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

H, W = 256, 256          # 影像尺寸
SEED = 42
OUT_SCENE = "data/scene.tif"
OUT_LABELS = "data/labels.tif"


def smooth_field(rng, scale):
    """生成一张 [0,1] 的空间平滑随机场:白噪声经高斯模糊 -> 低频、成片。"""
    noise = rng.standard_normal((H, W))
    field = gaussian_filter(noise, sigma=scale)
    field -= field.min()
    field /= (field.max() + 1e-9)
    return field


def main():
    rng = np.random.default_rng(SEED)

    # 1) 用两张平滑场决定地物分布,保证地物成片
    veg_field = smooth_field(rng, scale=12.0)   # 高处 -> 植被
    water_field = smooth_field(rng, scale=10.0)  # 高处 -> 水体

    labels = np.zeros((H, W), dtype=np.uint8)    # 默认 0 = 其他
    labels[veg_field > 0.60] = 1                 # 植被片区
    labels[water_field > 0.78] = 2               # 水体片区(占比更小,刻意做成少数类)

    # 2) 按每类的"典型光谱"生成 4 波段,再叠加噪声
    #    数值是 0-1 反射率风格,各类波段关系符合物理直觉:
    #      植被: NIR 高、Red 低 -> NDVI 高
    #      水体: NIR 低、Green 相对高 -> NDWI 高;SWIR 很低
    #      其他(裸土/建筑): 各波段中等偏高,较平
    # 每类的均值光谱: [Red, NIR, Green, SWIR]
    means = {
        0: np.array([0.30, 0.32, 0.28, 0.35]),  # 其他
        1: np.array([0.08, 0.55, 0.12, 0.22]),  # 植被
        2: np.array([0.06, 0.04, 0.10, 0.02]),  # 水体
    }
    noise_sigma = 0.04  # 类内噪声,制造一点重叠,避免完美可分

    scene = np.zeros((4, H, W), dtype=np.float32)
    for cls, mu in means.items():
        mask = labels == cls
        n = int(mask.sum())
        if n == 0:
            continue
        # 为该类的所有像素采样光谱
        samples = rng.normal(loc=mu, scale=noise_sigma, size=(n, 4)).astype(np.float32)
        for b in range(4):
            band = scene[b]
            band[mask] = samples[:, b]

    # 叠加一层轻微的、空间平滑的"地形阴影"扰动,让同类内部也有空间结构
    shading = (smooth_field(rng, scale=20.0) - 0.5) * 0.05
    scene = np.clip(scene + shading[None, :, :], 0.0, 1.0).astype(np.float32)

    # 3) 写出 GeoTIFF(带一个假的地理变换,够流程用)
    transform = from_origin(0, 0, 1, 1)
    with rasterio.open(
        OUT_SCENE, "w", driver="GTiff", height=H, width=W, count=4,
        dtype="float32", crs="EPSG:32650", transform=transform,
    ) as dst:
        dst.write(scene)
        for i, name in enumerate(["Red", "NIR", "Green", "SWIR"], start=1):
            dst.set_band_description(i, name)

    with rasterio.open(
        OUT_LABELS, "w", driver="GTiff", height=H, width=W, count=1,
        dtype="uint8", crs="EPSG:32650", transform=transform,
    ) as dst:
        dst.write(labels[None, :, :])

    # 4) 打印一下占比,方便核对
    total = labels.size
    print(f"写出 {OUT_SCENE} ({scene.shape}) 和 {OUT_LABELS} ({labels.shape})")
    for cls, name in [(0, "其他"), (1, "植被"), (2, "水体")]:
        c = int((labels == cls).sum())
        print(f"  类 {cls} ({name}): {c:>7d} 像素  占比 {c/total:6.2%}")


if __name__ == "__main__":
    main()
