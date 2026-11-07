## ABUS 二维拼接：离线 manifest 入口

本目录已在原 REW 实现上增加 `run_abus_pairwise.py`，可独立拷贝到服务器或 `vendor_baselines/python-Robust_Elastic_Warping-master/`。入口只调用本目录的 SIFT、单应矩阵 RANSAC、robust TPS 和平均融合，不下载模型或其他实现，无需训练。

先从 ABUS 数据目录生成 manifest（只扫描路径，不读取图像、不运行拼接）：

```bash
python python-Robust_Elastic_Warping-master/prepare_abus_manifest.py \
  --dataset-root /你的路径/dataset \
  --out-csv ./outputs/baseline_work/pairs_manifest.csv
```

支持 `dataset/case001/input1.jpg、input2.jpg、input3.jpg`，以及 `dataset/case001/input1/slice_0001.jpg` 等切片目录。自动生成 stage 12 和 23，优先同名切片；完全不同名且数量相同时按排序配对并提示。单图模式统一 slice 为 `0000`。生成的图像路径为绝对路径。存在 `nipple_x.txt` 时读取其三个坐标；缺少时 `left_x/right_x` 留空，REW 不使用这两个坐标。生成器只依赖 Python 标准库，不需要先安装 OpenCV。目标 CSV 已存在时拒绝覆盖。

安装及调用（以下命令供你在服务器上执行）：

```bash
pip install -r python-Robust_Elastic_Warping-master/requirements.txt
python python-Robust_Elastic_Warping-master/run_abus_pairwise.py \
  --manifest ./outputs/baseline_work/pairs_manifest.csv \
  --out-dir ./outputs/baselines/rew
```

`--manifest` 与你的离线对比协议相同，要求包含以下字段：

```csv
case,stage,slice,prefix,left,right,left_x,right_x
case001,12,0001,12_0001,/data/case001/input1/slice_0001.jpg,/data/case001/input2/slice_0001.jpg,256,240
case001,23,0001,23_0001,/data/case001/input2/slice_0001.jpg,/data/case001/input3/slice_0001.jpg,240,250
```

- 不扫描数据、不重新配对；单图及切片目录都通过 manifest 中的路径读取。
- `case/stage/slice/prefix` 来自 manifest；`prefix` 原样用于输出文件名，支持 stage `12` 和 `23`。
- `left/right` 推荐绝对路径；相对路径以 manifest 所在目录解析，不依赖启动目录。
- `left_x/right_x` 保留到诊断表和变换元数据中。原始 REW 通过 SIFT 对应点估计变换，不使用乳头坐标引导配准。
- 不 resize、不 crop，保持输入分辨率；为保留两幅图像的非重叠部分，输出采用 REW 扩展画布。支持左右不同尺寸。
- 读取方式与原代码一致：OpenCV 8 位 BGR；ABUS 体数据、高位深图像需先使用与主方法相同的二维预处理。

每行输出：

```text
<out-dir>/<stage>/<case>/
  warp/
    <prefix>_left.png
    <prefix>_right.png
    <prefix>_transform.json
  fusion/
    <prefix>_stitched.png
    <prefix>_mask_left_soft.png
    <prefix>_mask_right_soft.png
    <prefix>_mask_left_bin.png
    <prefix>_mask_right_bin.png
    <prefix>_seam_soft.png
    <prefix>_valid_left.png
    <prefix>_valid_right.png
```

`mask_*_soft` 是与实际拼接结果一致的融合权重：重叠区两侧各 0.5，单侧有效区该侧 1，画布外部 0；以 0–255 PNG 保存。`mask_*_bin` 按主项目的 `weight > 0.5` 规则保存，因此恰为 0.5 的平均融合重叠区两张二值 mask 都为 0。几何支撑范围单独存为 `valid_left/right`，有效黑色像素也计入。REW 没有优化接缝，`seam_soft` 在重叠区为 0.5、其他位置为 0，表示平均融合权重，不冒充接缝预测结果。

`rew_outputs.csv` 记录原始八个字段、状态、失败原因、耗时、匹配数和内点数；每对 `transform.json` 记录 left→right 单应矩阵和输入元数据。`rew_run_config.json` 记录运行参数。完整输入会在计算前检查：字段、文件存在性、stage、重复前缀及路径组件。已存在的目标结果拒绝覆盖，以免旧文件混入评价；允许调度器预先创建空输出目录。逐对计算失败时清理该对部分输出、继续处理后续行，最终返回退出码 1；不会用其他方法的结果代替失败。

可选参数：`--seed 0`、`--max-canvas-pixels 8000000`。像素上限只拒绝异常大画布，不缩放输入。

对接调度器时，REW 登记项的本地目录指向本目录，入口设为 `run_abus_pairwise.py`，命令模板为：

```text
python {entrypoint} --manifest {manifest} --out-dir {out_dir}
```

你提供的调度器原始登记表还没有 REW；仅提供命令模板不会自动新增 baseline key。当前工作区没有 `run_compare_baselines.py`，所以本次只实现本地入口，不假定已修改你的调度器登记表。

评价继续由主项目原有脚本执行，计算口径保持一致：

```bash
python eval_metrics.py --pairwise-root ./outputs/baselines/rew --out-csv ./outputs/rew_metrics.csv
python eval_pairwise_no_gt.py --pairwise-root ./outputs/baselines/rew --out-csv ./outputs/rew_no_gt.csv
```

本次只修改代码，不运行 ABUS 拼接或评价实验。原代码改动包括：可关闭 GUI、导出 warp 和几何掩膜、保护匹配不足/退化 TPS/超大画布、处理 `r² log(r²)` 在零处的极限和零残差融合。保留 SIFT、TPS 正则/网格间隔及平均融合策略；初始单应矩阵估计选项见下文。

---

# python-Robust_Elastic_Warping
Simple implementation of Robust Elastic Warping using opencv-python

*Parallax-Tolerant Image Stitching Based on Robust Elastic Warping* 这篇论文的Python实现，
仅实现了两幅拼接，需要的库：
```python
opencv-python==4.4.0.42
numpy
scipy
```

运行结果：
<table>
    <tr>
        <td ><center><img src="https://img-blog.csdnimg.cn/20200901212458335.png?x-oss-process" >mosaic_global</center></td>
        <td ><center><img src="https://img-blog.csdnimg.cn/20200901212738884.png?x-oss-process"  >mosaic_REW</center></td>
    </tr>
    <tr>
        <td ><center><img src="https://img-blog.csdnimg.cn/2020090121314848.jpg?x-oss-process" >mosaic_global</center></td>
        <td ><center><img src="https://img-blog.csdnimg.cn/20200901213147738.jpg?x-oss-process"  >mosaic_REW</center></td>
    </tr>
    <tr>
        <td ><center><img src="https://img-blog.csdnimg.cn/202009012129253.png?x-oss-process" >mosaic_global</center></td>
        <td ><center><img src="https://img-blog.csdnimg.cn/20200901212955734.png?x-oss-process"  >mosaic_REW</center></td>
    </tr>
    <tr>
        <td ><center><img src="https://img-blog.csdnimg.cn/20200901213324652.jpg?x-oss-process" >mosaic_global</center></td>
        <td ><center><img src="https://img-blog.csdnimg.cn/20200901213325812.jpg?x-oss-process"  >mosaic_REW</center></td>
    </tr>
</table>


## 多数样本报 Unbounded projected canvas 时

该错误表示单应矩阵的投影地平线穿过图像，或某角点靠近无穷远；保留拒绝处理，避免生成无界画布。矩阵检查现改为相对齐次分母判断，相同 H 乘任意非零比例不会改变结论，并检查正反方向。

ABUS 离线入口现在默认 `--homography-estimator opencv`：用 OpenCV RANSAC 和原始图像像素重投影误差替换原自定义 RANSAC 的代数误差评分，再接原 robust TPS。默认阈值 3 像素、最多 10000 次迭代、至少 6 个内点；重估后再次检查内点的实际误差及共线退化。不切换成仿射或其他方法来掩盖失败。

```bash
python python-Robust_Elastic_Warping-master/run_abus_pairwise.py \
  --manifest ./outputs/pairs_manifest.csv \
  --out-dir ./outputs/baselines/rew_pixel_ransac \
  --homography-estimator opencv \
  --ransac-reproj-threshold 3 \
  --ransac-max-iters 10000 \
  --min-inliers 6
```

保留 `--homography-estimator legacy --min-inliers 4` 复用旧版 HM_ransac 的 500 次、0.1 代数阈值。像素 RANSAC 是适配版的初始估计器改动，报告实验时须标明该配置，不能与旧估计器结果混为一组。legacy 模式不使用像素阈值和 max-iters 参数，画布有效性检查仍保留。

失败时额外保存 `diagnostics/<stage>/<case>/<prefix>_failure.json`，记录处理阶段、可用的匹配数/内点数、候选 H、图像尺寸、前 200 个内点对应坐标、重投影误差统计和错误原因。若在矩阵估计前失败则没有 H。`rew_outputs.csv` 保留失败时可用的数量，并指向诊断文件；成功元数据和运行配置也记录估计器。真正的错误匹配仍可能失败，本改动不保证真实数据成功率。

代码回归检查（仅数学与日志测试，不运行 ABUS 拼接实验）：

```bash
python -m pytest -q python-Robust_Elastic_Warping-master/tests/test_homography.py
```
