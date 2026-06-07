# 卫星影像植被分析工具 (NDVI)

从多波段卫星影像计算 NDVI、生成植被掩膜与统计报告的命令行工具。结果保留地理参考，可直接在 QGIS / ArcGIS 中使用。

## 文件说明

| 文件 | 说明 |
|------|------|
| `spec.md` | 规格文档（动手前的契约，含两个关键坑点的论证）|
| `ndvi_tool.py` | 工具本体（命令行）|
| `self_check.py` | 六项独立自检脚本 |
| `read_metadata.py` | 前期侦查：读影像元数据 |
| `analyze_edge_cases.py` | 前期侦查：验证「不转反射率会错多少」|
| `make_preview.py` | 生成三联预览图 + 物理交叉验证 |
| `input.tif` | 输入影像（2 波段 uint16，Red/NIR）|
| `output_ndvi.tif` | 输出：NDVI 栅格（float32，nodata=-9999）|
| `output_vegetation_mask.tif` | 输出：植被掩膜（1=植被/0=非植被/255=无效）|
| `statistics.txt` | 输出：统计报告 |
| `preview.png` | 三联预览图 |

## 运行环境

- Python 3 + rasterio + numpy + matplotlib

```bash
pip install rasterio numpy matplotlib
```

## 用法

```bash
python ndvi_tool.py input.tif [--red 1] [--nir 2] [--threshold 0.2] [--outdir .]
```

## 自检

```bash
python self_check.py
```

六项自检全部通过：手算对拍、独立重读波段、范围检查 [-1,1]、边界情况（填充/除零）、地理参考一致、掩膜一致。

## 两个关键坑点

1. **必须先转反射率再算 NDVI**：本影像 `OFFSET=-0.2 != 0`，偏移量在 NDVI 分母里不约分。用原始 DN 直算误差可达 0.35，且结果仍落在 [-1,1] 看不出错。正确公式：`reflectance = DN * 2.75e-05 + (-0.2)`。
2. **uint16 下溢**：计算前必须 `.astype('float64')`，否则 `NIR-Red` 为负时整型下溢。

详见 `spec.md`。
