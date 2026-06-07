"""
train_eval.py
-------------
植被/水体像素级分类:随机森林 + 两种切分方式对比。

本脚本的重点不是"训练一个模型",而是**诚实地评估它**:
  1. 先算多数类基线(不会犯错的"傻"参照)。
  2. 不只看 accuracy,打印混淆矩阵 + 每类 precision/recall/F1 + macro-F1。
  3. 同一个模型、同一套特征,分别用【随机像素切分】和【空间区块切分】评估,
     把两组分数并排放出来 —— 这就是"测试集有没有偷看训练集"的实证。
  4. 输出预测图,肉眼检查空间合理性。

用法:
  python src/train_eval.py                       # 默认读 data/scene.tif, data/labels.tif
  python src/train_eval.py --scene path --labels path --out report
"""

import argparse
import os
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             f1_score, balanced_accuracy_score)
from scipy.ndimage import distance_transform_edt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = {0: "其他", 1: "植被", 2: "水体"}
CLASS_NAMES_EN = {0: "other", 1: "veg", 2: "water"}   # 出图用,避免 matplotlib 缺中文字体
EPS = 1e-6


# --------------------------------------------------------------------------- #
# 1. 读数据 + 造特征
# --------------------------------------------------------------------------- #
def load_data(scene_path, labels_path):
    with rasterio.open(scene_path) as s:
        scene = s.read().astype(np.float32)      # (4, H, W): Red, NIR, Green, SWIR
        profile = s.profile
    with rasterio.open(labels_path) as l:
        labels = l.read(1).astype(np.int64)      # (H, W)
    assert scene.shape[0] >= 4, "期望至少 4 个波段: Red/NIR/Green/SWIR"
    return scene, labels, profile


def build_features(scene):
    """从 4 波段拼出特征立方体 (n_features, H, W)。

    原始 4 波段 + NDVI + NDWI + MNDWI。指数是领域先验,把植被/水体的物理含义
    直接喂给模型,而不是让模型从原始波段里自己摸索。
    """
    red, nir, green, swir = scene[0], scene[1], scene[2], scene[3]
    ndvi = (nir - red) / (nir + red + EPS)          # 植被 -> 高(NIR 强反射、Red 强吸收)
    ndwi = (green - nir) / (green + nir + EPS)       # 水体 -> 高(水对 NIR 强吸收)
    mndwi = (green - swir) / (green + swir + EPS)     # 水体 -> 高(用 SWIR 替 NIR,对含建成区/阴影的影像辨水更稳)
    feats = np.stack([red, nir, green, swir, ndvi, ndwi, mndwi], axis=0)
    names = ["Red", "NIR", "Green", "SWIR", "NDVI", "NDWI", "MNDWI"]
    return feats, names


# --------------------------------------------------------------------------- #
# 2. 两种切分
# --------------------------------------------------------------------------- #
def split_random(n_pixels, test_frac=0.3, seed=0):
    """随机像素切分:打乱所有像素索引再切。

    ⚠️ 错误示范:相邻像素几乎一样,随机切会让测试集塞满训练集的空间近邻 -> 泄漏。
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_pixels)
    cut = int(n_pixels * (1 - test_frac))
    return idx[:cut], idx[cut:]


def split_spatial_blocks(H, W, n_blocks=4, test_frac=0.3, seed=0):
    """空间区块切分:把影像切成 n_blocks×n_blocks 个网格块,整块分给 train 或 test。

    测试块与训练块在空间上分离,相邻像素不会被拆到两边 -> 测试集是模型没见过的区域。
    返回展平后(行优先)的 train_idx, test_idx。
    """
    rng = np.random.default_rng(seed)
    rows = np.array_split(np.arange(H), n_blocks)
    cols = np.array_split(np.arange(W), n_blocks)

    block_ids = []                       # 每个块: (r_slice, c_slice)
    for r in rows:
        for c in cols:
            block_ids.append((r, c))

    n_total = len(block_ids)
    n_test = max(1, int(round(n_total * test_frac)))
    perm = rng.permutation(n_total)
    test_blocks = set(perm[:n_test].tolist())

    # 用一张块归属图标记每个像素属于 train(0) 还是 test(1)
    assign = np.zeros((H, W), dtype=bool)   # True = test
    for bi, (r, c) in enumerate(block_ids):
        if bi in test_blocks:
            assign[np.ix_(r, c)] = True

    flat = assign.reshape(-1)
    all_idx = np.arange(H * W)
    return all_idx[~flat], all_idx[flat]


# --------------------------------------------------------------------------- #
# 3. 评估:不止 accuracy
# --------------------------------------------------------------------------- #
def majority_baseline(y_train, y_test, classes):
    """多数类基线:永远预测训练集里占比最大的那一类。"""
    vals, counts = np.unique(y_train, return_counts=True)
    major = vals[np.argmax(counts)]
    y_pred = np.full_like(y_test, fill_value=major)
    acc = (y_pred == y_test).mean()
    macro = f1_score(y_test, y_pred, labels=classes, average="macro", zero_division=0)
    return major, acc, macro


def evaluate(name, y_true, y_pred, classes):
    """打印一种切分下的完整评估,并返回汇总 dict。"""
    acc = (y_true == y_pred).mean()
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)
    macro = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)

    print(f"\n===== 评估:{name} =====")
    print(f"总体 accuracy      = {acc:.4f}   (仅供参考,会被多数类抬高)")
    print(f"balanced accuracy = {bal_acc:.4f}   (各类 recall 的平均,小类失败会立刻拉低它)")
    print(f"macro-F1          = {macro:.4f}   (类别等权,对不平衡更诚实)")
    print("\n混淆矩阵 (行=真值, 列=预测):")
    header = "          " + "".join(f"{CLASS_NAMES[c]:>8}" for c in classes)
    print(header)
    for i, c in enumerate(classes):
        row = "".join(f"{cm[i, j]:>8d}" for j in range(len(classes)))
        print(f"真值 {CLASS_NAMES[c]:<4}{row}")
    print("\n每类指标:")
    print(f"{'类别':<6}{'precision':>11}{'recall':>9}{'F1':>9}{'support':>10}")
    for i, c in enumerate(classes):
        print(f"{CLASS_NAMES[c]:<6}{p[i]:>11.3f}{r[i]:>9.3f}{f[i]:>9.3f}{sup[i]:>10d}")

    return {"name": name, "acc": acc, "bal_acc": bal_acc, "macro_f1": macro, "cm": cm,
            "precision": p, "recall": r, "f1": f, "support": sup}


def plot_confusion(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([CLASS_NAMES_EN[c] for c in classes])
    ax.set_yticklabels([CLASS_NAMES_EN[c] for c in classes])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. 主流程
# --------------------------------------------------------------------------- #
def run_one_split(tag, train_idx, test_idx, X, y, classes):
    """在给定切分上:训练 RF -> 评估,并打印多数类基线作对照。"""
    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]

    major, base_acc, base_macro = majority_baseline(ytr, yte, classes)
    print(f"\n#### 切分方式:{tag} ####")
    print(f"训练像素 {len(train_idx)}, 测试像素 {len(test_idx)}")
    print(f"[多数类基线] 全猜「{CLASS_NAMES[major]}」-> accuracy={base_acc:.4f}, "
          f"macro-F1={base_macro:.4f}")

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1,
        class_weight="balanced", random_state=0)
    clf.fit(Xtr, ytr)
    y_pred = clf.predict(Xte)

    res = evaluate(f"{tag} / 随机森林", yte, y_pred, classes)
    res["baseline_acc"] = base_acc
    res["baseline_macro_f1"] = base_macro
    res["clf"] = clf
    return res


def leakage_distance_diagnostic(train_idx, test_idx, H, W):
    """量化"测试集到底离训练集多近"——这是数据泄漏的结构性证据。

    即使两种切分准确率都一样,只要随机切分下"测试像素紧贴训练像素",
    它在结构上就是不可信的:模型不需要理解光谱,照抄隔壁邻居的标签就能拿高分。

    做法:把训练像素标记成前景,做欧氏距离变换,再在测试像素位置取距离值,
    得到每个测试像素到最近训练像素的距离分布。
    """
    train_mask = np.zeros(H * W, dtype=bool)
    train_mask[train_idx] = True
    dist2d = distance_transform_edt(~train_mask.reshape(H, W))
    test_mask = np.zeros(H * W, dtype=bool)
    test_mask[test_idx] = True
    d = dist2d[test_mask.reshape(H, W)]
    return {"min": float(d.min()), "median": float(np.median(d)),
            "mean": float(d.mean()), "max": float(d.max()),
            "frac_le_1": float((d <= 1).mean())}


def run_weak_feature_experiment(X, y, classes, tr_r, te_r, tr_s, te_s, feat_names):
    """弱特征对照实验:故意只用 1 个波段(Red),把模型削弱,让小类失败暴露出来。

    主流水线用 7 维强特征时,这景数据完全可分,两种切分都 100%,
    "高准确率掩盖小类失败"那一课演示不出来。这里把特征削到只剩 Red,
    模型会被迫犯错——而它犯错的方式正是任务警告的:总体 accuracy 还很高,
    但占比最小的水体 recall 崩掉,total accuracy 把这个失败藏了起来。
    这就是"为什么 99% 准确率可能毫无用处"的活例子。
    """
    print("\n\n############ 弱特征对照实验(只用 Red 一个波段)############")
    print("目的:把模型削弱,演示「总体 accuracy 高、但小类 recall 崩」的假象。\n")
    Xw = X[:, [0]]   # 只取 Red
    out = {}
    for tag, tr, te in [("弱特征 / 随机切分", tr_r, te_r),
                        ("弱特征 / 空间切分", tr_s, te_s)]:
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                     n_jobs=-1, class_weight="balanced", random_state=0)
        clf.fit(Xw[tr], y[tr])
        res = evaluate(tag, y[te], clf.predict(Xw[te]), classes)
        out[tag] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data/scene.tif")
    ap.add_argument("--labels", default="data/labels.tif")
    ap.add_argument("--out", default="report")
    ap.add_argument("--n-blocks", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scene, labels, profile = load_data(args.scene, args.labels)
    feats, feat_names = build_features(scene)
    n_feat, H, W = feats.shape

    X = feats.reshape(n_feat, -1).T          # (n_pixels, n_features)
    y = labels.reshape(-1)
    classes = sorted(np.unique(y).tolist())

    # ---- 各类占比(用于判断多数类基线)----
    print("====== 数据概况 ======")
    total = y.size
    for c in classes:
        cnt = int((y == c).sum())
        print(f"类 {c} ({CLASS_NAMES.get(c, c)}): {cnt:>8d}  占比 {cnt/total:6.2%}")

    # ---- 两种切分 ----
    tr_r, te_r = split_random(total, test_frac=0.3, seed=0)
    res_random = run_one_split("随机像素切分(会泄漏)", tr_r, te_r, X, y, classes)

    tr_s, te_s = split_spatial_blocks(H, W, n_blocks=args.n_blocks, test_frac=0.3, seed=0)
    res_spatial = run_one_split("空间区块切分(诚实)", tr_s, te_s, X, y, classes)

    # ---- 对比小结 ----
    print("\n\n############ 两种切分对比 ############")
    print(f"{'指标':<18}{'随机切分':>14}{'空间切分':>14}")
    print(f"{'accuracy':<18}{res_random['acc']:>14.4f}{res_spatial['acc']:>14.4f}")
    print(f"{'balanced acc':<18}{res_random['bal_acc']:>14.4f}{res_spatial['bal_acc']:>14.4f}")
    print(f"{'macro-F1':<18}{res_random['macro_f1']:>14.4f}{res_spatial['macro_f1']:>14.4f}")
    drop = res_random['acc'] - res_spatial['acc']
    print(f"\naccuracy 下降 {drop:.4f} —— 若为 0,说明这景数据在强特征下完全可分,")
    print("泄漏的影响被特征强度盖住了(并不代表随机切分就可信,见下面的距离诊断)。")

    # ---- 数据泄漏的结构性证据:测试像素离训练像素有多近 ----
    print("\n\n############ 数据泄漏结构诊断:测试像素到最近训练像素的距离 ############")
    print("回答「你怎么保证测试集没在偷看训练集」——即使准确率一样,结构也能分高下。\n")
    d_random = leakage_distance_diagnostic(tr_r, te_r, H, W)
    d_spatial = leakage_distance_diagnostic(tr_s, te_s, H, W)
    print(f"{'切分':<14}{'min':>7}{'median':>9}{'mean':>8}{'max':>7}{'紧贴(<=1px)占比':>16}")
    for nm, d in [("随机切分", d_random), ("空间切分", d_spatial)]:
        print(f"{nm:<14}{d['min']:>7.2f}{d['median']:>9.2f}{d['mean']:>8.2f}"
              f"{d['max']:>7.2f}{d['frac_le_1']*100:>14.2f}%")
    print(f"\n解读:随机切分下 {d_random['frac_le_1']*100:.1f}% 的测试像素紧贴训练像素"
          "(距离<=1px),")
    print("即测试集几乎坐在训练集身上——模型照抄邻居标签就能拿高分,这是结构性泄漏。")
    print(f"空间切分把整块分到一边,紧贴比例降到 {d_spatial['frac_le_1']*100:.1f}%"
          f",中位距离 {d_spatial['median']:.0f}px,测试区是模型没见过的区域。")

    # ---- 弱特征对照实验:演示「99% 准确率却没用」----
    weak = run_weak_feature_experiment(X, y, classes, tr_r, te_r, tr_s, te_s, feat_names)
    wk = list(weak.values())[1]   # 弱特征 / 空间切分
    water_idx = classes.index(2) if 2 in classes else -1
    print("\n>>> 关键看点:弱特征下总体 accuracy 仍然不低,但水体 recall 崩了 <<<")
    for tag, res in weak.items():
        wr = res['recall'][water_idx] if water_idx >= 0 else float('nan')
        print(f"  {tag}: accuracy={res['acc']:.4f}, balanced_acc={res['bal_acc']:.4f}, "
              f"macro-F1={res['macro_f1']:.4f}, 水体 recall={wr:.3f}")
    print("→ 这就是「为什么 99% 准确率可能毫无用处」:总体分被多数类抬高,")
    print("  而你真正在乎的水体几乎全没认出来,只有 balanced_acc / macro-F1 / 水体 recall 戳穿它。")

    # ---- 特征重要性(验证模型真在用 NDVI/NDWI/MNDWI)----
    imp = res_spatial["clf"].feature_importances_
    print("\n特征重要性(空间切分模型):")
    for nm, v in sorted(zip(feat_names, imp), key=lambda t: -t[1]):
        print(f"  {nm:<6}{v:.3f}")

    # ---- 用空间切分的模型对全图预测,输出 prediction.tif ----
    full_pred = res_spatial["clf"].predict(X).reshape(H, W).astype(np.uint8)
    pred_profile = profile.copy()
    pred_profile.update(count=1, dtype="uint8")
    pred_path = os.path.join(args.out, "prediction.tif")
    with rasterio.open(pred_path, "w", **pred_profile) as dst:
        dst.write(full_pred[None, :, :])

    # ---- 出图:混淆矩阵 + 预测图 vs 真值图 ----
    plot_confusion(res_random["cm"], classes, "random split",
                   os.path.join(args.out, "cm_random.png"))
    plot_confusion(res_spatial["cm"], classes, "spatial split",
                   os.path.join(args.out, "cm_spatial.png"))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    cmap = matplotlib.colors.ListedColormap(["#cccccc", "#2ca02c", "#1f77b4"])
    axes[0].imshow(labels, cmap=cmap, vmin=0, vmax=2); axes[0].set_title("ground truth")
    axes[1].imshow(full_pred, cmap=cmap, vmin=0, vmax=2); axes[1].set_title("prediction")
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "map_pred_vs_truth.png"), dpi=120)
    plt.close(fig)

    print(f"\n已写出:{pred_path}, cm_random.png, cm_spatial.png, map_pred_vs_truth.png")


if __name__ == "__main__":
    main()
