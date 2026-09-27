"""Build the editable Word edition of the ABUS stitching manuscript.

The implementation deliberately uses only Python's standard library so that the
checked-in document can be reproduced in restricted environments.  Equations
are emitted as native Office Math (OMML) objects and tables as native Word
tables; neither is rasterised.
"""

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).with_name("ABUS_two_stage_stitching_word.xml")

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def run(text, bold=False, italic=False):
    props = ("<w:rPr>" + ("<w:b/>" if bold else "")
             + ("<w:i/>" if italic else "") + "</w:rPr>") if bold or italic else ""
    return f'<w:r>{props}<w:t xml:space="preserve">{escape(str(text))}</w:t></w:r>'


def paragraph(text="", style=None, bold=False, center=False):
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if center:
        ppr.append('<w:jc w:val="center"/>')
    return f'<w:p><w:pPr>{"".join(ppr)}</w:pPr>{run(text, bold=bold)}</w:p>'


def equation(text, number=None):
    suffix = f"    ({number})" if number is not None else ""
    return (f'<m:oMathPara><m:oMath><m:r><m:t>{escape(text + suffix)}</m:t>'
            '</m:r></m:oMath></m:oMathPara>')


def table(headers, rows, widths=None):
    def cell(value, head=False):
        shade = '<w:shd w:fill="D9EAF7"/>' if head else ""
        return (f'<w:tc><w:tcPr>{shade}</w:tcPr><w:p><w:pPr><w:jc w:val="center"/>'
                f'</w:pPr>{run(value, bold=head)}</w:p></w:tc>')
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in (widths or [1800] * len(headers)))
    head = '<w:tr>' + ''.join(cell(x, True) for x in headers) + '</w:tr>'
    body = ''.join('<w:tr>' + ''.join(cell(x) for x in row) + '</w:tr>' for row in rows)
    return ('<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/>'
            '</w:tblPr><w:tblGrid>' + grid + '</w:tblGrid>' + head + body + '</w:tbl>')


parts = [
    paragraph("基于结构先验的多视图 ABUS 两阶段无监督图像拼接", "Title", center=True),
    paragraph("Word 可编辑版本（公式为 Office Math，表格为原生 Word 表格）", center=True),
    paragraph("1  Introduction", "Heading1"),
    paragraph("超声成像具有无放射、非侵入、实时和低成本等优势。自动化三维乳腺超声（ABUS）能够降低手持扫描对操作者经验的依赖，但受探头尺寸限制，单侧乳房通常需采集外侧位（LAT）、前后位（AP）和内侧位（MED）。因此，将具有重叠区域的多视图图像拼接为宽视野图像具有重要意义。"),
    paragraph("传统方法依赖 SIFT、SURF、ORB 等人工特征与 RANSAC 变换估计；超声图像的低对比度、散斑噪声和软组织形变使这些特征不够稳定。深度学习能够学习更鲁棒的表示，但针对多视图 ABUS 的工作仍较少。本文提出基于乳头结构先验的两阶段无监督框架：首先完成全局单应与局部 TPS 配准，再估计软接缝和逐像素融合权重。"),
    paragraph("主要贡献", "Heading2"),
    paragraph("（1）利用三维 ABUS 中乳头的 z 坐标初始化冠状面纵向位置，并利用 x 坐标构造结构先验；（2）以共享多尺度编码器、全局单应和局部 TPS 学习复杂形变；（3）在无人工配准真值条件下，以图像自身的一致性训练配准与融合网络。"),
    paragraph("2  Related works", "Heading1"),
    paragraph("2.1  Non-learning based image stitching methods", "Heading2"),
    paragraph("传统流程由特征提取、匹配、几何估计和融合组成。APAP、半投影形变、全局相似性先验、点线一致性、弹性形变和接缝优化分别从局部几何、整体形状和视觉连续性改善自然图像拼接。针对超声，已有方法结合超像素、轮廓角点、块匹配、3D SIFT、传感器导航和体数据无缝融合，但通常仍依赖人工特征、显式优化或预定义模型。"),
    paragraph("2.2  Learning-based image stitching methods", "Heading2"),
    paragraph("无监督深度拼接逐步将特征、变换和融合变为可学习模块，并由全局单应扩展到 TPS、像素级形变与边界规整。超声领域已有胎儿三维复合、SynStitch、LOTUS、DFMH-Net、VFMStitch 等工作。ABUS 研究则更多关注病灶检测、定位和单视图重建，LAT、AP、MED 原始数据的连续配准与自然融合仍有不足。"),
    paragraph("3  Method", "Heading1"),
    paragraph("原始输入为不同扫描视图的三维 ABUS 体数据。首先依据乳头 z 坐标沿 y 方向粗对齐冠状面；随后将图像输入两阶段网络。左图 I_L 为源图像，右图 I_R 为固定参考图像。"),
    equation("I_L^w = GridSample(I_L, G_final),    I_R^w = I_R", 1),
    paragraph("3.1  3D Spatial Initialization", "Heading2"),
    paragraph("乳头 z 坐标提供不同体数据在切片方向上的初始位置；粗初始化后，二维网络估计精细变换。乳头 x 坐标进一步参与无监督损失和扩展画布定位，形成三维初始化与二维精配准的两级约束。"),
    paragraph("3.2  Coarse-to-Fine Geometric Alignment", "Heading2"),
    paragraph("共享 ResNet-50 编码器输出 1/16 与 1/8 尺度特征。粗阶段在相同空间位置计算归一化特征相关性，经 FCA 聚合并回归四角点偏移；DLT 据此计算单应矩阵。"),
    equation("C(x,y) = Σ_(c=1)^(C_f) F̂_L^(1/16)(c,x,y) F̂_R^(1/16)(c,x,y)", 2),
    equation("H_c = DLT(P, P + ΔP)", 3),
    paragraph("局部阶段预测 9×9 控制点位移，以 TPS 平滑插值，降低逐像素位移的自由度并保持空间连续性。"),
    equation("Δ(p) = a_0 + a_x p_x + a_y p_y + Σ_(i=1)^N w_i U(‖p-c_i‖_2),    U(r)=r² ln(r²+ε)", 4),
    equation("G_final = G_H + Δ,    G_H(x)=π(H_c^(-1) x̃)", 5),
    paragraph("3.2.1  ABUS Structural Prior", "Heading3"),
    paragraph("以乳头横向坐标为中心构造沿图像高度复制的高斯权重。左右响应的逐点最大值形成 H_nip；它表示乳头所在纵向带状区域，而非二维关键点距离。"),
    equation("H_x(x,y) = exp(-(x-x_nip)²/(2σ²))", 6),
    paragraph("3.2.2  Alignment Loss Function", "Heading3"),
    equation("L_warp = λ_1 L_L1 + λ_e L_edge + λ_a L_angle + λ_n L_nipple", 7),
    equation("L_L1 = [Σ_p Ω(p)|I_L^w(p)-I_R^w(p)|]/[Σ_p Ω(p)+ε]", 8),
    equation("L_edge = mean|‖e^x‖_2-l_x^0| + mean|‖e^y‖_2-l_y^0|", 9),
    equation("L_angle = mean |(e^x·e^y)/(‖e^x‖_2 ‖e^y‖_2+ε)|", 10),
    equation("L_nipple = (1/HW) Σ_p H_nip(p) mean_c |I_L^w(p)-I_R^w(p)|", 11),
    paragraph("3.3  Seam-aware Image Fusion", "Heading2"),
    paragraph("由非零区域获得有效掩膜及重叠区域 Ω。在 Ω 内计算均值和标准差，将参考图像统计分布匹配到源图像，并通过重叠掩膜局部平均得到平滑权重 α，使强度校正主要作用于重叠区附近。"),
    equation("I_R^m = α[(I_R^w-μ_R) σ_L/(σ_R+ε)+μ_L] + (1-α)I_R^w", 12),
    equation("D = |I_L^w-I_R^m|", 13),
    paragraph("U-Net 型网络使用膨胀率 1、2、4 的空洞卷积，并将多尺度差分特征注入解码器跳跃连接。输出经 1×1 卷积和 Sigmoid 得到 soft seam，经 mask refinement 获得归一化权重。"),
    equation("S_soft = sigmoid(f_seam(d_1))", 14),
    equation("I_stitch = M_L ⊙ I_L^w + M_R ⊙ I_R^m", 15),
    paragraph("3.3.1  Fusion Loss Function", "Heading3"),
    equation("L_fusion = λ_s L_smooth + λ_seam L_seam + λ_ncc L_NCC", 16),
    equation("L_smooth = (1/HW) Σ_p (|∇_x I_stitch(p)|+|∇_y I_stitch(p)|)", 17),
    equation("L_seam = [Σ_(p∈B_seam)|I_L^w(p)-I_R^w(p)|]/[|B_seam|+ε]", 18),
    equation("L_NCC = 1-NCC(S,R)", 19),
    paragraph("4  Experiments and Results", "Heading1"),
    paragraph("4.1  Experimental Setup", "Heading2"),
    paragraph("数据规模、训练/验证/测试划分、优化器、初始学习率、batch size、epoch 数及 GPU 型号在原稿中仍以 XXX 占位，需在投稿前补充。各方法使用相同测试图像对和评价区域。"),
    paragraph("4.2  Evaluation Metrics", "Heading2"),
    paragraph("在 Overlap region 评价共同组织的空间对齐，在 Seam-band region 评价融合边界连续性。采用 PSNR、SSIM、NCC 与 MSE；前三者越高越好，MSE 越低越好，表中 MSE 以 ×10⁻² 报告。Seam-band 宽度仍需补充。"),
    paragraph("表 1  与传统拼接方法的定量比较", bold=True),
]

traditional = [
    ["APAP","Overlap","18.2838","0.8072","0.8299","10.3744"],["APAP","Seam-band","15.6639","0.5453","0.5817","25.0717"],
    ["Identity","Overlap","18.4195","0.8224","0.8412","9.7070"],["Identity","Seam-band","15.7846","0.5565","0.5938","23.3924"],
    ["ORB","Overlap","18.3647","0.8191","0.8388","9.9271"],["ORB","Seam-band","15.7096","0.5535","0.5907","23.8918"],
    ["SIFT","Overlap","18.2821","0.8112","0.8323","10.2923"],["SIFT","Seam-band","15.5306","0.5453","0.5818","25.0504"],
    ["SPC","Overlap","23.8917","0.9203","0.9244","3.3365"],["SPC","Seam-band","18.6392","0.7962","0.7993","10.1799"],
    ["SPW","Overlap","18.3519","0.8113","0.8321","10.1501"],["SPW","Seam-band","15.7080","0.5493","0.5852","24.3751"],
    ["Ours","Overlap","25.8735","0.9421","0.9494","1.9101"],["Ours","Seam-band","19.3104","0.8338","0.8488","8.0505"],
]
parts.append(table(["Method","Region","PSNR ↑","SSIM ↑","NCC ↑","MSE (×10⁻²) ↓"], traditional))
parts += [paragraph("本文方法在 Overlap 四项指标上均最优；相对 SPC，PSNR 提升 1.9818 dB，MSE 从 3.3365×10⁻² 降至 1.9101×10⁻²。在 Seam-band 中也取得传统方法中的最佳结果。"), paragraph("表 2  与学习式拼接方法的定量比较", bold=True)]
deep = [
 ["UDH","Overlap","21.4908","0.8638","0.8509","7.4123"],["UDH","Seam-band","19.1482","0.6812","0.6313","16.7992"],
 ["UDIS++","Overlap","21.9370","0.8645","0.8782","7.1310"],["UDIS++","Seam-band","20.0517","0.5934","0.5669","38.2514"],
 ["DH","Overlap","18.5590","0.7227","0.6865","15.7635"],["DH","Seam-band","20.5685","0.7107","0.6558","12.4232"],
 ["Ours","Overlap","25.8735","0.9421","0.9494","1.9101"],["Ours","Seam-band","19.3104","0.8338","0.8488","8.0505"],
]
parts.append(table(["Method","Region","PSNR ↑","SSIM ↑","NCC ↑","MSE (×10⁻²) ↓"], deep))
parts += [paragraph("本文方法在 Overlap 全部指标最佳。在 Seam-band，DH 的 PSNR 较高，但本文方法的 SSIM、NCC 和 MSE 最优，说明其更好地保持组织结构与灰度变化一致性。"), paragraph("4.3  Ablation Study", "Heading2"), paragraph("表 3  编码器与预训练策略消融", bold=True)]
enc = [
 ["ResNet50","ImageNet","Overlap","25.8735","0.9421","0.9494","1.9101"],["ResNet50","ImageNet","Seam-band","19.3104","0.8338","0.8488","8.0505"],
 ["ResNet50","RadImageNet","Overlap","21.9196","0.8462","0.8541","4.5119"],["ResNet50","RadImageNet","Seam-band","19.5635","0.6724","0.6994","9.7925"],
 ["DenseNet121","ImageNet","Overlap","21.9496","0.8476","0.8555","4.4773"],["DenseNet121","ImageNet","Seam-band","19.6781","0.6767","0.7034","9.5274"],
 ["DenseNet121","RadImageNet","Overlap","21.8961","0.8456","0.8537","4.5258"],["DenseNet121","RadImageNet","Seam-band","19.7257","0.6748","0.7019","9.5854"],
 ["InceptionV3","ImageNet","Overlap","21.9809","0.8479","0.8561","4.4548"],["InceptionV3","ImageNet","Seam-band","19.5944","0.6746","0.7017","9.6605"],
 ["InceptionV3","RadImageNet","Overlap","22.0545","0.8442","0.8577","4.3393"],["InceptionV3","RadImageNet","Seam-band","19.6657","0.6701","0.6926","9.4309"],
]
parts.append(table(["Encoder","Pre-training","Region","PSNR ↑","SSIM ↑","NCC ↑","MSE ↓"], enc))
parts += [paragraph("ResNet50 + ImageNet 在 Overlap 综合最佳，并在 Seam-band 的 SSIM、NCC 和 MSE 上最佳，因此作为默认配置。"), paragraph("表 4  乳头结构先验与 NCC 损失消融", bold=True)]
loss = [
 ["w/o Nipple","Overlap","20.9036","0.8401","0.8497","5.6727"],["w/o Nipple","Seam-band","19.8479","0.7339","0.7649","9.0950"],
 ["w/o NCC","Overlap","25.2239","0.8863","0.8911","2.8849"],["w/o NCC","Seam-band","20.1893","0.6952","0.6950","13.2081"],
 ["Full","Overlap","25.8735","0.9421","0.9494","1.9101"],["Full","Seam-band","19.3104","0.8338","0.8488","8.0505"],
]
parts.append(table(["Configuration","Region","PSNR ↑","SSIM ↑","NCC ↑","MSE (×10⁻²) ↓"], loss))
parts += [
 paragraph("去除乳头先验使 Overlap PSNR 从 25.8735 降至 20.9036；去除 NCC 后 Seam-band 的 SSIM、NCC 明显下降且 MSE 上升。完整模型在几何对齐与融合连续性之间最均衡。"),
 paragraph("5  Discussion", "Heading1"),
 paragraph("该方法虽然输出二维冠状面拼接，却利用三维 ABUS 的乳头坐标辅助配准。当前仍以 pairwise stitching 为基本任务，尚未显式约束 LAT、AP、MED 的整体空间一致性。后续可在统一乳腺坐标系中联合估计三视图关系，并进一步扩展到完整体数据的三维形变、融合与跨设备验证。多模态配准、术前规划与导航仍属于尚未验证的研究方向。"),
 paragraph("6  Conclusion", "Heading1"),
 paragraph("本文提出基于结构先验的两阶段无监督 ABUS 拼接方法，以全局单应和 TPS 补偿几何变化，以乳头位置约束多视图配准，并以接缝感知网络完成自适应融合。实验及消融结果表明，结构先验改善空间对齐，NCC 约束有助于保持融合区域结构与灰度一致性。未来将研究三维、多视图联合建模及跨域泛化。"),
 paragraph("References", "Heading1"),
]

refs = [
"Boca et al. Pros and cons for automated breast ultrasound (ABUS): A narrative review. Journal of Personalized Medicine, 2021.",
"Fischler and Bolles. Random sample consensus. Communications of the ACM, 1981.",
"Brown and Lowe. Automatic panoramic image stitching using invariant features. IJCV, 2007.",
"Zaragoza et al. As-projective-as-possible image stitching with moving DLT. CVPR, 2013.",
"Che, Mathai, and Galeotti. Ultrasound registration: A review. Methods, 2017.",
"Yan et al. Larynx ultrasound image stitching based on multiconstraint super-pixel feature. IEEE TIM, 2023.",
"Jiang et al. Ultrasound image stitching fusion based on contour corner points. SIVP, 2025.",
"Wright et al. Complete fetal head compounding from multi-view 3DUS. MICCAI, 2019.",
"Wright et al. Fast fetal head compounding from multi-view 3D ultrasound. Medical Image Analysis, 2023.",
"Yao et al. SynStitch. IEEE ISBI, 2025.", "Yao et al. LOTUS. MIDL, 2026.",
"Yan et al. Deep feature masking and homograph for cranial and larynx ultrasound image stitching. JUM, 2026.",
"Yao et al. VFMStitch. MIDL, 2026.", "Lowe. Object recognition from local scale-invariant features. ICCV, 1999.",
"Bay et al. Speeded-up robust features (SURF). CVIU, 2008.", "Rublee et al. ORB. ICCV, 2011.",
"Chang et al. Shape-preserving half-projective warps. CVPR, 2014.", "Chen and Chuang. Natural image stitching with the global similarity prior. ECCV, 2016.",
"Jia et al. Leveraging line-point consistence. CVPR, 2021.", "Du et al. Geometric structure preserving warp. CVPR, 2022.",
"Liao and Li. Natural image stitching using depth maps. arXiv:2202.06276, 2023.", "Zhang and Liu. Parallax-tolerant image stitching. CVPR, 2014.",
"Gao et al. Seam-driven image stitching. Eurographics, 2013.", "Lin et al. SEAGULL. ECCV, 2016.",
"Li et al. Parallax-tolerant image stitching based on robust elastic warping. IEEE TMM, 2018.",
"Li et al. Perception-based seam cutting. SIVP, 2018.", "Liao et al. Quality evaluation-based iterative seam estimation. SIVP, 2019.",
"Peng et al. Seamless UAV hyperspectral image stitching. IEEE TGRS, 2023.", "Banerjee et al. Fast and robust 3D ultrasound registration. Medical Image Analysis, 2015.",
"Freesmeyer et al. Stitching of sensor-navigated 3D ultrasound datasets. Medical Ultrasonography, 2018.",
"Seifert et al. Optimization of thyroid volume determination by stitched 3D ultrasound. Biomedicines, 2023.",
"Ni et al. Volumetric ultrasound panorama based on 3D SIFT. MICCAI, 2008.", "Flach et al. PURE. Springer, 2016.",
"Gomez et al. Fast registration of 3D fetal ultrasound images. Springer, 2017.", "Chang et al. Rapid image stitching for multipass ABUS. Medical Physics, 2010.",
"Nie et al. Unsupervised deep image stitching. IEEE TIP, 2021.", "Nie et al. Parallax-tolerant unsupervised deep image stitching. ICCV, 2023.",
"Jia et al. Learning pixel-wise alignment. ACM MM, 2023.", "Kweon et al. Pixel-wise warping for deep image stitching. AAAI, 2023.",
"Nie et al. Deep rectangling for image stitching. CVPR, 2022.", "Xie et al. Reconstructing the image stitching pipeline. NeurIPS, 2024.",
"Gomez et al. Whole-fetus ultrasound imaging by patch manifolds. Springer, 2019.", "Yao et al. From geometry to intensity. SPIE, 2026.",
"章浩伟等. 基于自动乳腺超声的肿瘤定位与三维重建系统. 北京生物医学工程, 2024.",
"曾焕城, 陈嘉炜. 基于 YOLOv5 模型的自动乳腺超声乳头目标检测. 中国医疗器械信息, 2024.",
"耿如霞等. 一种自动化三维乳腺超声全景图的自动拼接算法. 电子测量技术, 2021.",
]
parts.extend(paragraph(f"[{i}] {ref}") for i, ref in enumerate(refs, 1))

document = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W}" xmlns:m="{M}"><w:body>{''.join(parts)}
<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>
</w:body></w:document>'''

styles = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W}">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="宋体"/><w:sz w:val="21"/></w:rPr><w:pPr><w:spacing w:line="360" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:i/><w:sz w:val="22"/></w:rPr></w:style>
<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/><w:bottom w:val="single" w:sz="4"/><w:right w:val="single" w:sz="4"/><w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/></w:tblBorders></w:tblPr></w:style>
</w:styles>'''

content_types = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>'''
rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''
doc_rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''

def without_declaration(xml):
    """Remove the declaration before embedding an XML part in Flat OPC."""
    return xml.split("?>", 1)[1].lstrip()


# Flat OPC is Word's single-file, XML representation of a .docx package. It is
# used as the checked-in deliverable because review systems can display it,
# unlike a binary .docx. Word preserves the native equations and tables.
flat_opc = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<pkg:package xmlns:pkg="http://schemas.microsoft.com/office/2006/xmlPackage">
  <pkg:part pkg:name="/_rels/.rels" pkg:contentType="application/vnd.openxmlformats-package.relationships+xml">
    <pkg:xmlData>{without_declaration(rels)}</pkg:xmlData>
  </pkg:part>
  <pkg:part pkg:name="/word/document.xml" pkg:contentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml">
    <pkg:xmlData>{without_declaration(document)}</pkg:xmlData>
  </pkg:part>
  <pkg:part pkg:name="/word/styles.xml" pkg:contentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml">
    <pkg:xmlData>{without_declaration(styles)}</pkg:xmlData>
  </pkg:part>
  <pkg:part pkg:name="/word/_rels/document.xml.rels" pkg:contentType="application/vnd.openxmlformats-package.relationships+xml">
    <pkg:xmlData>{without_declaration(doc_rels)}</pkg:xmlData>
  </pkg:part>
</pkg:package>'''
OUT.write_text(flat_opc.replace("><", ">\n<"), encoding="utf-8")
print(OUT)
