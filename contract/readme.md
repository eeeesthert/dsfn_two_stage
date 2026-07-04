## 离线对比实验适配

> 重要说明：本仓库不会假装已经复现或内置下列 GitHub 仓库的完整代码。若服务器不能访问 GitHub，需要你先通过其他方式把对应实现拷贝到本地；本项目提供的是**离线输入适配与本地运行入口**，保证这些本地实现接收与主模型完全一致的 ABUS 两两输入。

已登记的对比方法与默认本地目录：

| key | 方法 | 默认本地目录 | 原始参考 |
| --- | --- | --- | --- |
| `udhr2018` | unsupervisedDeepHomographyRAL2018 | `vendor_baselines/unsupervisedDeepHomographyRAL2018` | https://github.com/tynguyen/unsupervisedDeepHomographyRAL2018 |
| `deep_homography` | DeepHomography | `vendor_baselines/DeepHomography` | https://github.com/JirongZhang/DeepHomography |
| `udis2` | UDIS2 | `vendor_baselines/UDIS2` | https://github.com/nie-lang/UDIS2 |
| `pixelstitch` | PixelStitch | `vendor_baselines/PixelStitch` | https://github.com/MakiseChris666/PixelStitch |

适配逻辑与主模型一致：自动按 `case*/input1-input2` 生成 stage `12`，按 `input2-input3` 生成 stage `23`，支持单图和切片目录两种数据组织；同时导出统一的 `pairs_manifest.csv`，字段为 `case/stage/slice/prefix/left/right/left_x/right_x`。

只准备统一输入和适配配置：

```bash
python run_compare_baselines.py \
  --dataset-root ./dataset \
  --work-dir ./outputs/baseline_work \
  --prepare-only
```

每个本地 baseline 目录下默认需要提供一个 `run_abus_pairwise.py`，它至少支持：

```bash
python run_abus_pairwise.py --manifest /path/to/pairs_manifest.csv --out-dir /path/to/output_root
```

该脚本应读取 manifest 中的 `left/right` 图像路径，并输出与本项目一致的结构：`<out-dir>/<stage>/<case>/warp/` 与 `<out-dir>/<stage>/<case>/fusion/`，包括 `*_left.png`、`*_right.png`、`*_stitched.png`、mask 和 seam 文件。

如果本地代码已放在 `vendor_baselines/` 下，可运行：

```bash
python run_compare_baselines.py \
  --dataset-root ./dataset \
  --vendor-root ./vendor_baselines \
  --out-dir ./outputs/baselines
```

若某个本地实现的命令行参数不同，可覆盖命令模板：

```bash
python run_compare_baselines.py \
  --dataset-root ./dataset \
  --baselines udis2 pixelstitch \
  --command 'udis2=python {entrypoint} --pairs {manifest} --save_dir {out_dir}' \
  --command 'pixelstitch=python {entrypoint} --manifest {manifest} --output {out_dir}'
```

支持的占位符包括 `{entrypoint}`、`{manifest}`、`{out_dir}`、`{baseline}`、`{work_dir}`、`{impl_dir}`。如果没有提供对应本地代码，脚本会明确报错，不会生成伪造的对比结果。
