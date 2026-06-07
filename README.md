# GIS + AI 任务集

两个遥感影像处理任务,各自独立、规格先行、自带自检。

| 任务 | 内容 | 目录 |
|------|------|------|
| 任务一 | **NDVI 植被分析工具** —— 从多波段影像算 NDVI、生成植被掩膜与统计,保留地理参考 | [`task1-ndvi/`](task1-ndvi/) |
| 任务二 | **植被/水体像素级分类器** —— 随机森林分类,重点是"看不出对错时怎么判断模型可不可信" | [`task2-classification/`](task2-classification/) |

## 任务一:NDVI 工具

从 2 波段(Red/NIR)卫星影像计算 NDVI,输出 NDVI 栅格、植被掩膜和统计报告。
重点是两个容易踩的坑:**先转反射率再算 NDVI**(偏移量不约分)、**uint16 下溢**(算前转 float64)。
详见 [task1-ndvi/README.md](task1-ndvi/README.md)。

## 任务二:植被/水体分类

4 波段影像上的像素级分类(其他/植被/水体)。不追求高 accuracy,追求**诚实评估**:
多数类基线、混淆矩阵、每类 precision/recall、两种切分对比、泄漏结构诊断、弱特征对照实验。
核心问题:**为什么一个准确率 99% 的模型可能毫无用处。**
详见 [task2-classification/README.md](task2-classification/README.md) 与
[task2-classification/report/REPORT.md](task2-classification/report/REPORT.md)。

## 运行

每个任务目录可独立运行,环境都是 Python 3 + rasterio + numpy + scikit-learn + matplotlib。
进入对应目录按各自 README 操作即可。
