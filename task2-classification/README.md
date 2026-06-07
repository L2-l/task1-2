# 植被/水体遥感分类(GIS + AI)

任务二:训练一个像素级模型,把一景 4 波段遥感影像里的 **植被 / 水体 / 其他** 分出来。
重点不是刷精度,而是 **当你没法肉眼判断对错时,怎么用对的方式相信(或不相信)一个模型**。

> 核心收获:**为什么一个准确率 99% 的模型可能毫无用处?**
> 因为高分全压在多数类(植被 73%)上,占比 1.66% 的水体它可能几乎全没分对,而总体准确率会把这个失败藏起来。
> 见 [report/REPORT.md](report/REPORT.md)。

## 结构

```
SPEC.md                     规格(目标/方法/切分/指标/验收/非目标)
src/make_synthetic_data.py  生成有空间连续性的合成影像+标签(无真数据时跑通流程用)
src/train_eval.py           读数据 -> 造特征(7维:4波段+NDVI/NDWI/MNDWI)-> 两种切分
                            -> 评估 + 泄漏结构诊断 + 弱特征对照 -> 出图
report/REPORT.md            评估报告:多数类基线、混淆矩阵、各类 P/R/F1、两种切分对比
report/run_output.txt       train_eval.py 在真数据上的完整原始输出
report/*.png                混淆矩阵图 + 预测 vs 真值图
data/                       放 scene.tif / labels.tif(不入库)
```

## 环境

```
python >= 3.10
numpy, scipy, scikit-learn, rasterio, matplotlib
```

安装:`pip install -r requirements.txt`

## 跑起来

```bash
# 1) 没有真数据时,先生成合成数据(写到 data/)
python src/make_synthetic_data.py

# 2) 训练 + 评估 + 出图(结果写到 report/)
python src/train_eval.py

# 拿到导师真数据后,直接指向真文件:
python src/train_eval.py --scene path/to/scene.tif --labels path/to/labels.tif --out report
```

波段顺序约定:Band1=Red, Band2=NIR, Band3=Green, Band4=SWIR;标签 0=其他,1=植被,2=水体。

## 自检清单(对应任务要求)

- [x] 先算**多数类基线**,确认模型显著超过它
- [x] 不只看 accuracy:打印**混淆矩阵 + 每类 precision/recall/F1 + balanced accuracy + macro-F1**
- [x] **两种切分对比**:随机像素切分(会泄漏)vs 空间区块切分(诚实)
- [x] **泄漏结构诊断**:测试像素到最近训练像素的距离分布(即使准确率一样也能分高下)
- [x] **弱特征对照实验**:演示"总体 accuracy 高、但水体 recall 崩"的假象
- [x] **空间合理性**:预测图叠看,水体是否成片、有没有椒盐噪声
- [x] 能回答:"你怎么保证测试集没在偷看训练集?"(见 REPORT §3.2)
