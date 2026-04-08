# DSFN Two-Stage ABUS Pairwise Stitching

本仓库实现了你要求的 **ABUS 两两视图拼接**（先 1-2，再 2-3），按 DSFN 的两阶段思想：

1. **Warp 阶段（两阶段）**：
   - 阶段1：1/16 特征上做 FCA + RR，回归四顶点偏移，DLT 求粗单应矩阵 \(H_c\)
   - 阶段2：预测网格控制点位移，使用高斯 RBF 插值连续变形场 \(\Delta\)
   - 最终 warp：`img ∘ Hc + Δ`
2. **Fusion 阶段**：dilated conv U-Net，在 skip 连接中引入差分特征，预测 overlap 内 soft seam，并与 warp mask 融合得到最终 mask

> FOV 说明：fusion 输入使用 warp 输出的扩展画布，不固定为原图宽度；会根据左右乳头 x 位置扩展并重定位，尽量使乳头位于输出 x 中心，同时保留两侧非重叠区域。

损失函数：
- Warp 阶段：`overlap_l1_warp_loss`、`grid_edge_length_loss`、`grid_angle_loss`、`nipple_heatmap_alignment_loss`
- Fusion 阶段：`seam_overlap_boundary_loss`、`seam_cost_loss`、`fusion_smoothness_loss`、`nipple_heatmap_alignment_loss`
- 优化器：Adam + ExponentialLR
- 默认损失权重（重叠区域较大时推荐）：`warp_l1=1.0, grid_edge=4.0, grid_angle=2.0, warp_nipple=0.5, seam_boundary=1.0, seam_cost=2.0, fusion_smooth=0.2, fusion_nipple=0.5`

## 数据格式

```
./dataset/case001/input1.jpg
./dataset/case001/input2.jpg
./dataset/case001/input3.jpg
./dataset/case001/nipple_x.txt   # [x1,x2,x3]
```

也支持切片目录格式（你现在这个格式）：

```
./dataset/case001/input1/slice_0001.jpg
./dataset/case001/input2/slice_0001.jpg
./dataset/case001/input3/slice_0001.jpg
./dataset/case001/nipple_x.txt
```

程序会优先按同名 `slice_xxx` 对齐，不同名时按排序后一一配对。

## 训练（两步）

```bash
python train_pairwise.py --dataset-root ./dataset --out-dir ./outputs
```

可选训练增强与防过拟合：
- `--augment` 开启数据增强（成对同步增强）
- `--hflip-prob`、`--brightness-jitter`、`--contrast-jitter`
- `--val-split` + `--early-stopping-patience` 启用 early stopping 与 best checkpoint（每个 stage 保存 `stage_xx_best.pt`）

可插拔预训练来源（编码器）：
- `--encoder-pretrain-source imagenet`：ImageNet 预训练（默认）
- `--encoder-pretrain-source radimagenet --encoder-ckpt /path/to/radimagenet_resnet50.pth`
- `--encoder-pretrain-source local --encoder-ckpt /path/to/selfsup_checkpoint.pth`
- `--encoder-pretrain-source none`：随机初始化（不加载预训练）

`radimagenet` 支持两种方式：
1) 显式提供 `--encoder-ckpt` 本地权重；  
2) 设置环境变量 `RADIMAGENET_RESNET50_URL` 自动下载。  
若两者都不提供，会自动回退到 ImageNet 权重并提示。

也可以直接在命令行里传（无需手动设置环境变量）：

```bash
python train_pairwise.py --encoder-pretrain-source radimagenet --radimagenet-url https://your-url/radimagenet_resnet50.pth
```

等价写法（兼容参数名）：

```bash
python train_pairwise.py --encoder-pretrain-source radimagenet --net-url https://your-url/radimagenet_resnet50.pth
```

默认使用交叉训练（interleaved）并导出：
- 每个 epoch 交替优化 `input1 + input2` (`stage=12`) 与 `input2 + input3` (`stage=23`)
- 两步共享同一套网络参数，训练结束仅保存一个共享 checkpoint（默认 `shared_model.pt`）

如果你想回到“先 12 再 23”的顺序训练，可加：

```bash
python train_pairwise.py --dataset-root ./dataset --training-schedule sequential
```

每一步都会保存：
- `warp/`：warp 后 left/right 图像
- `fusion/`：融合结果 + soft mask + 二值 mask（每张图像保留区域）

默认会根据 warp 后有效区域（重叠+非重叠并集）自动裁剪输出，因此每个 case 的输出尺寸可以不同，不再固定为 512×512。
如果你希望保留原始切片分辨率，不要 resize，设置：

```bash
python train_pairwise.py --dataset-root ./dataset --image-size 0
```

## 推理

```bash
python infer_pairwise.py --dataset-root ./dataset --checkpoint ./outputs/shared_model.pt --out-dir ./infer_outputs
```

同样支持 `--image-size 0` 保留输入原始尺寸。

如果 checkpoint 是基于本地自监督编码器训练得到，推理时请保持相同编码器来源参数，例如：

```bash
python infer_pairwise.py \
  --dataset-root ./dataset \
  --checkpoint ./outputs/shared_model.pt \
  --encoder-pretrain-source local \
  --encoder-ckpt /path/to/selfsup_checkpoint.pth
```

如果希望推理后直接做三视图融合（推荐）：

```bash
python infer_pairwise.py \
  --dataset-root ./dataset \
  --checkpoint ./outputs/shared_model.pt \
  --out-dir ./infer_outputs \
  --run-three-view-fusion
```

## 三视图融合（基于 input2 在 12/23 的 mask）

有了 `12` 和 `23` 的结果后，可以运行：

```bash
python fuse_three_view.py --pairwise-root ./outputs/results --out-dir ./outputs/three_view --levels 5 --input2-boost 2.0
```

融合策略：
- 读取 stage12/stage23 的 `fusion/*_stitched.png`
- 使用 `stage12` 的 `mask_right` 和 `stage23` 的 `mask_left` 作为 input2 权重来源
- 在 input2 的重叠区域（两张 mask 共同高响应处）放大 input2 权重
- 用高斯/拉普拉斯金字塔进行三图加权融合
- 以 case 为单位统一尺寸：取该 case 最小尺寸，其他切片做中心左右裁剪 + 底部裁剪
- 再对融合结果做 CLAHE（先裁剪再增强）
- 输出 `threeview_xxx.png`，并保存评价指标 `metrics.csv`（配准一致性 + 融合一致性）

## 说明

- 目前先完成你要求的 **二维两两拼接**。
- 已提供基于 step1/step2 的 input2 mask 的三视图高斯金字塔融合脚本（见上节）。
