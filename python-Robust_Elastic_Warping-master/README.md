## ABUS 二维拼接：离线 manifest 入口

本目录已在原 REW 实现上增加 `run_abus_pairwise.py`，可独立拷贝到服务器或 `vendor_baselines/python-Robust_Elastic_Warping-master/`。入口只调用本目录的 SIFT、HM_ransac、robust TPS 和平均融合，不下载模型或其他实现，无需训练。

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

本次只修改代码，不运行 ABUS 拼接或评价实验。原代码改动包括：可关闭 GUI、导出 warp 和几何掩膜、保护匹配不足/退化 TPS/超大画布、处理 `r² log(r²)` 在零处的极限和零残差融合。保留 SIFT、原 RANSAC、TPS 正则/网格间隔及平均融合策略。

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

