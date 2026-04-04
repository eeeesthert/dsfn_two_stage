# DSFN Two-Stage ABUS Pairwise Stitching

本仓库实现了你要求的 **ABUS 两两视图拼接**（先 1-2，再 2-3），按 DSFN 的两阶段思想：

1. **Warp 阶段**：ResNet50 多尺度编码 + 两步回归（全局 x shift + dense flow）
2. **Fusion 阶段**：dilated conv U-Net 预测 soft seam mask 做融合

并将原始深度先验/损失替换为 **乳头 x 坐标强先验**：
- `nipple_prior_loss`：约束预测全局平移与 `(x_right - x_left)` 对齐
- `x_heatmap_similarity_loss`：只基于 x 坐标构建热图，强调乳头周围区域拼接相似性

## 数据格式

```
./dataset/case001/input1.jpg
./dataset/case001/input2.jpg
./dataset/case001/input3.jpg
./dataset/case001/nipple_x.txt   # [x1,x2,x3]
```

## 训练（两步）

```bash
python train_pairwise.py --dataset-root ./dataset --out-dir ./outputs
```

会依次训练并导出：
- 第一步 `input1 + input2` (`stage=12`)
- 第二步 `input2 + input3` (`stage=23`)

每一步都会保存：
- warp 后图像
- fusion 结果
- mask

## 推理

```bash
python infer_pairwise.py --dataset-root ./dataset --checkpoint ./outputs/stage_23.pt --out-dir ./infer_outputs
```

## 说明

- 目前先完成你要求的 **二维两两拼接**。
- 三视图高斯金字塔融合还没做，但输出里已经保存了 step1/step2 的 mask，可作为后续 input2 重叠区融合先验。
