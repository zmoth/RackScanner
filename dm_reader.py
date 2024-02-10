import os
import logging
import time
from functools import partial

import numpy as np
import cv2
from pylibdmtx import pylibdmtx
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy, scipy.ndimage, scipy.stats


statistics = pd.DataFrame()  # 初始化统计DataFrame，用于存储识别过程中的统计数据
failed = pd.DataFrame()  # 初始化失败记录DataFrame，用于存储识别失败的冻存管信息

# 定义解码方法列表，包含不同类型的识别方式
methods = ["empty", "raw", "lsd", "harris", "unchanged", "rotated", "failed"]

# 定义 DataMatrix 二维码读取参数字典，设置最大检测数量、超时时间等参数
libdmtx_params = dict(
    max_count=1,
    timeout=300,
    min_edge=10,
    max_edge=100,
    threshold=5,
    deviation=10,
    # shape = 1 #DataMatrix.DmtxSymbol12x12 (注释掉的参数表示使用默认的符号大小)
)

well_size = 150  # 设置单个冻存管图像尺寸为150x150像素
well_shape = (150, 150)  # 设置冻存管裁剪区域形状（宽和高）
well_center = (75, 75)  # 计算冻存管中心点坐标
# 创建一个孔洞模板，用以帮助识别冻存管内孔洞位置
peephole = cv2.circle(np.zeros(well_shape), well_center, 30, 1, -1)
min_size = 65  # 设置最小可接受的冻存管尺寸阈值
dm_size = None  # 初始化动态尺寸变量dm_size，用于存储实际检测到的冻存管尺寸
dg_img = None  # 初始化全局变量dg_img，用于存储原图数据
n_wells = None  # 初始化全局变量n_wells，用于存储当前图像中检测到的冻存管数量

# 检查日志文件是否存在，如果不存在则创建并写入表头
if not os.path.exists("dm_reader_log.csv"):
    with open("dm_reader_log.csv", "w") as f:
        f.write(
            ", ".join(["timestamp", "ms", "level", "filename", "duration"] + methods)
            + "\n"
        )

# 配置日志记录器，将输出内容保存至 dm_reader_log.csv 文件中
logging.basicConfig(
    filename="dm_reader_log.csv",
    format="%(asctime)s, %(levelname)s, %(message)s",
    level=logging.INFO,
)


def read(filename: str, vial=False, debug=False):
    """
    read 函数用于加载指定图像文件，识别其中的冻存管排布并读取所有底码信息。

    Args:
        filename (str): 待处理图像文件路径。
        vial (bool, optional): 是否仅识别单个冻存管。默认为 False（识别整个冻存管盒）。
        debug (bool, optional): 是否开启调试模式以显示中间过程和结果。默认为 False。

    Returns:
        _type_: 返回一个包含每个冻存管位置、编码及解码方法等信息的 DataFrame，同时返回缩放后的原图。

    Raises:
        Exception: 如果无法打开指定图像文件，则抛出异常。

    实现流程：
    1. 加载指定图像，并将其转换为灰度图像。
    2. 调用 `locate_wells` 函数定位所有冻存管的位置。
    3. 使用 `partial` 函数创建 `read_well` 函数的部分应用实例，传入全局图像数据。
    4. 对于每个识别到的冻存管位置，应用 `read_well_partial` 函数读取底码信息。
    5. 处理识别失败的情况，将失败的记录保存至全局变量 `failed` 中。
    6. 计算识别耗时并统计不同解码方法的成功次数。
    7. 将统计数据追加至全局变量 `statistics`。
    8. 在调试模式下，显示处理过的原图像。

    最终返回包含所有冻存管信息的 DataFrame 和缩放后的原图。
    """

    global dg_img

    # 开始计时
    start = time.time()

    # 加载图像并转为灰度图，进行转置操作以便在 notebook 中查看效果
    img = cv2.imread(filename, 0)
    if img is None:
        raise Exception('Cannot open image "%s"' % filename)
    dg_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # 定位冻存管位置
    wells = locate_wells(img, vial=vial)
    if wells is None:
        return None

    # 创建部分应用函数，用于读取每个冻存管的底码
    read_well_partial = partial(read_well, img=img)

    # 应用函数获取每个冻存管的详细信息
    wells = wells.apply(read_well_partial, axis=1)

    # 处理识别失败的冻存管记录
    fail = wells.loc[wells.method == "failed"].copy()
    if not fail.empty:
        fail["well"] = fail.apply(lambda x: get_well_matched(img, x), axis=1)
        fail["file"] = filename
        global failed
        failed = pd.concat([failed, fail], axis=0)

    # 计算识别耗时
    duration = time.time() - start

    # 统计不同解码方法的成功次数
    stats = wells.groupby("method").size()
    stats = stats.reindex(methods).fillna(0).astype(int)
    global statistics
    statistics = statistics._append(stats, ignore_index=True)

    # 构建包含文件名、耗时及解码统计信息的数据序列
    stats = (
        pd.Series((filename, duration), ["filename", "duration"])
        ._append(stats)
        .astype(str)
    )

    # 输出日志信息
    logging.info(", ".join(list(stats)))

    # 调试模式下显示原图
    if debug:
        plt.imshow(dg_img)
        plt.show()

    # 返回所有冻存管信息的 DataFrame 及缩放后的原图
    return wells, dg_img


def read_well(coo, img):
    """
    read_well 函数用于识别传入坐标（coo）对应位置上的单个冻存管底部的条形码或二维码信息。

    Args:
        coo: 包含x和y坐标的对象，表示待识别冻存管在图像中的位置。
        img (_type_): 输入的图像数据，其中包含了要读取的冻存管底码。

    Returns:
        pandas.Series: 返回一个包含以下信息的 Series 对象：
            - x (float): 冻存管在图像中的 x 坐标位置。
            - y (float): 冻存管在图像中的 y 坐标位置。
            - code (str): 解码得到的条形码或二维码内容。
            - method (str): 识别所采用的方法（例如 'barcode' 或 'qr_code'）。

    实现流程：
    1. 使用 `get_well_matched` 函数根据坐标 `coo` 从原图 `img` 中裁剪出目标冻存管区域。
    2. 调用 `read_barcode` 函数对裁剪出来的冻存管区域进行底码识别，并返回识别到的编码内容以及识别方法。
    3. 使用 `mark_well` 函数标记已成功识别的冻存管（可能用于后续处理或可视化显示）。
    4. 将识别结果以指定格式封装为 Pandas Series 并返回。
    """

    # 获取与给定坐标匹配的冻存管图像区域
    well = get_well_matched(img=img, coo=coo)

    # 识别并读取条形码或二维码
    code, method = read_barcode(well)

    # 标记该冻存管已经完成识别
    mark_well(coo, method)

    # 将识别结果包装为一个 Pandas Series
    return pd.Series([coo.x, coo.y, code, method], index=["x", "y", "code", "method"])


def locate_wells(img, vial=False, debug=False):
    """
    locate_wells 函数用于定位图像中冻存管盒或单个试管的排列顺序，通过模板匹配方法找到每个试管槽的位置，并将其转换为一个 Pandas DataFrame 表格形式返回。

    Args:
        img (_type_): 输入的灰度图像数据。
        vial (bool, optional): 是否仅识别单个试管。默认为 False（识别96孔或24孔板）。
        debug (bool, optional): 是否开启调试模式以显示中间处理结果。默认为 False。

    Returns:
        pandas.DataFrame: 返回一个包含每个试管位置信息（行、列坐标）的数据框。

    实现流程：
    1. 根据 `vial` 参数决定是识别单个试管还是整块板子。
    2. 如果是识别单个试管，则利用 Harris 角点检测算法找到试管中心并调整坐标。
    3. 否则，使用模板匹配方法对整个板子进行匹配，根据匹配到的井数量确定板子类型（96孔或24孔），并计算出每个孔洞的位置。
    4. 计算每个孔洞的质心坐标，并根据孔洞布局生成对应的行和列标签。
    5. 将所有坐标信息整理到一个 Pandas DataFrame 中，按照行和列排序后设置索引。
    """

    # 使用全局变量 dm_size 和 n_wells
    global dm_size
    global n_wells

    if vial:
        # 单个试管识别
        n_wells, n_rows, n_cols, dm_size = 1, 1, 1, [12, 14]
        harris = cv2.cornerHarris(img, 4, 1, 0.0)
        thr = threshold(harris, 0.1)
        arr = np.round(scipy.ndimage.center_of_mass(thr)).astype(int) - np.array(
            [75, 75]
        )
        arr = np.expand_dims(arr, axis=0)

    else:
        # 整块板子识别
        labeled, n_wells, crop = matchTemplate(
            img, "resources/template_96.png", debug=debug
        )
        b = 100

        # 判断是否匹配到96孔板
        if n_wells == 96:
            n_wells, n_rows, n_cols, dm_size, origin = (
                96,
                8,
                12,
                [12],
                np.array([35, 40]),
            )
        else:
            # 若未匹配到96孔板，则尝试匹配24孔板
            labeled, m, crop = matchTemplate(
                img, "resources/template_24.png", debug=debug
            )

            # 判断是否匹配到24孔板
            if m == 24:
                n_wells, n_rows, n_cols, dm_size, origin = (
                    24,
                    4,
                    6,
                    [14],
                    np.array([150, 150]),
                )
            else:
                # 提示错误信息并返回 None
                print("%s and %s wells detected. Should be 24 or 96." % (n_wells, m))
                return None

        # 计算匹配区域的质心并调整坐标
        arr = (
            np.round(
                scipy.ndimage.center_of_mass(crop, labeled, range(1, n_wells + 1))
            ).astype(int)
            + b
            + origin
        )

    # 创建 DataFrame 存储结果
    df = pd.DataFrame(arr, columns=("y", "x"))

    # 添加行标签（字母 A-H）
    LETTERS = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    df["row"] = LETTERS[np.arange(n_rows).repeat(n_cols)]

    # 对 DataFrame 进行排序并添加列标签
    df = df.sort_values(["row", "x"])
    df["col"] = np.tile(np.arange(1, n_cols + 1), n_rows)

    # 设置多级索引
    df = df.set_index(["row", "col"], drop=True)

    return df


def matchTemplate(img, templ_file, debug=False):
    """
    matchTemplate 函数实现模板匹配算法，将指定的模板图片与原始图像进行匹配，并在原图中找到相似度最高的区域。

    Args:
        img (_type_): 原始图像数据，通常为灰度图像或单通道彩色图像的 numpy 数组。
        templ_file (_type_): 模板图片文件路径，用于加载作为模板的图像数据。
        debug (bool, optional): 是否开启调试模式以显示中间处理结果。默认为 False。

    Returns:
        _type_: 返回以下内容组成的元组：
            - labeled: 通过标记后的图像数组，其中每个连通区域都用不同的整数标签表示。
            - n_wells: 匹配到的井（或目标区域）的数量。
            - crop: 经过阈值处理后保留显著匹配结果的部分。

    实现流程：
    1. 加载模板图片并提取其第一个通道作为模板数据。
    2. 使用 OpenCV 的 cv2.matchTemplate 方法计算模板与原始图像间的相关系数归一化匹配结果。
    3. 对匹配结果进行裁剪，去除边缘可能存在的噪声影响。
    4. 应用阈值操作来过滤掉低匹配度的结果。
    5. 如果调试模式开启，则展示原始图像、匹配结果和经过阈值处理后的图像。
    6. 使用 scipy.ndimage.label 函数对阈值处理后的结果进行标记，返回标记后的图像以及统计的连通区域数量。
    """

    # 加载模板图片并转换为灰度图像
    template = cv2.imread(templ_file)[:, :, 0]

    # 计算模板与原始图像之间的匹配度
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

    # 裁剪匹配结果以减少边缘效应的影响
    b = 100
    crop = res[b:-b, b:-b]

    # 应用阈值过滤低匹配度结果
    th, crop = cv2.threshold(crop, 0.6, 1, cv2.THRESH_TOZERO)

    # 在调试模式下显示图像
    if debug:
        plt.subplot(131)
        plt.imshow(img, cmap="gray")
        plt.title("Original Image")

        plt.subplot(132)
        plt.imshow(res, cmap="gray")
        plt.title("Match Template Result")

        plt.subplot(133)
        plt.imshow(crop, cmap="gray")
        plt.title("Thresholded Match Result")

        plt.show()

    # 标记匹配结果中的各个连通区域
    labeled, n_wells = scipy.ndimage.label(crop)

    return labeled, n_wells, crop


def get_well_matched(img, coo):
    """
    get_well_matched 函数用于从输入图像 img 中裁剪出一个以坐标 coo 为中心的 150x150 大小的区域，并将裁剪后的子区域进行转置，最后创建一个新的按 C 风格（行主序）顺序排列的副本。

    Args:
        img (numpy.ndarray): 原始图像数据，通常为 OpenCV 或 Numpy 处理的多维数组（通道数 x 高度 x 宽度）。
        coo (_type_): 坐标对象，包含中心点的横纵坐标信息，coo.x 和 coo.y 分别表示该点在图像中的列索引和行索引。

    Returns:
        numpy.ndarray: 返回一个形状为 (150, 150, channels) 的三维数组（对于彩色图像，channels=3；对于灰度图像，channels=1），这个数组是原始图像中指定区域经过转置并按 C 语言风格内存布局复制得到的新数组。

    注意：实际应用时，请确保输入的坐标不会超出图像边界。
    """

    # 裁剪出图像中对应位置的 150x150 区域
    cropped_region = img[coo.y : coo.y + 150, coo.x : coo.x + 150]

    # 创建按 C 风格顺序排列的副本
    c_style_copy = cropped_region.copy(order="C")

    return c_style_copy


#     ox, oy, dx, dy = (200, 160, 212, 206)
#     py = oy + row * dy
#     px = ox + col * dx
#     well = img[py:py+200, px:px+200]
#     well[0:5, :] = 0; well[:, 0:5] = 0 # for diagnostics
#     return well[5:, 5:].copy(order = 'F')


def mark_well(coo, mark):
    """
    mark_well 函数用于在全局变量 dg_img 所代表的图像上，根据给定坐标 coo 以及标记类型 'mark' 绘制不同颜色和样式的圆圈。

    Args:
        coo (_type_): 坐标对象或元组，包含点的 x 和 y 坐标信息。
        mark (str): 标记字符串，表示要绘制的圆圈的不同含义，可选值包括 "raw", "lsd", "harris", "unchanged", "rotated", "failed", "empty" 等。

    注：此函数假设有一个全局变量 dg_img 存储了待处理的图像数据，该图像应为 OpenCV 能够处理的 numpy 数组格式（如灰度图或彩色图像）。

    实现过程：
    1. 根据输入的标记类型从预定义的颜色字典中获取相应的 BGR 颜色元组。
    2. 使用 OpenCV 的 cv2.circle 函数，在 dg_img 图像的指定位置 (coo.x + 75, coo.y + 75) 绘制一个半径为 75 的圆圈，并设置指定的颜色和线条粗细为 10 像素。
    """

    # 定义颜色对应表，根据不同的标记类型返回对应的 BGR 颜色
    color_dict = {
        "raw": (0, 255, 0),
        "lsd": (150, 150, 0),
        "harris": (255, 255, 0),
        "unchanged": (50, 100, 0),
        "rotated": (100, 100, 0),
        "failed": (255, 0, 0),
        "empty": (0, 0, 0),
    }

    # 获取与标记类型关联的颜色
    color = color_dict[mark]

    # 访问并操作全局图像变量
    global dg_img

    cv2.putText(
        dg_img, mark, (coo.x + 75, coo.y + 75), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5
    )

    # 在图像上以给定坐标为中心绘制圆圈
    cv2.circle(dg_img, (coo.x + 75, coo.y + 75), 75, color=color, thickness=10)


def read_barcode(well):
    """
    read_barcode 函数用于读取并识别给定图像 `well` 中的条形码或二维码信息。

    Args:
        well (_type_): 一个二维图像数组，通常为灰度图像，表示单个孔板井（例如微孔板）中的区域，可能包含条形码或二维码。

    Returns:
        _type_: 返回一个包含两个元素的元组。第一个元素是解码得到的字符串（条形码或二维码的内容），第二个元素是标记类型字符串，描述了所使用的解码方法或状态。

    此函数执行以下步骤：
    1. 对输入图像应用阈值处理以二值化图像，并与预定义的“窥孔”掩模进行乘法操作以优化条形码检测区域。
    2. 检查二值化后的图像是否为空（像素总和小于500），如果是，则返回 "empty" 标记。

    接着尝试使用不同的解码策略：
    3. 使用 `decode_raw` 函数尝试直接解码原始图像，如果成功则返回解码内容及 "raw" 标记。

    （注：这里省略了对 LSD 算法的尝试，若启用可恢复该部分代码）

    4. 应用 Harris 角点检测算法增强后，使用 `decode_harris` 函数尝试解码，成功时返回解码内容及 "harris" 标记。

    5. 如果上述所有优化手段都未能成功解码，则使用基础的 `decode` 函数尝试解码未经改进的原图。
      成功时返回解码内容及 "unchanged" 标记。

    6. 若仍无法解码，则通过傅里叶变换进行图像旋转校正，然后再次尝试解码。
      若旋转校正后成功解码，则返回解码内容及 "rotated" 标记。

    7. 若所有尝试均失败，则返回 "failed" 标记，表示无法从图像中成功解码出条形码或二维码信息。
    """
    # 阈值处理，将图像二值化以便于后续处理
    x, thr = cv2.threshold(well, 128, 1, cv2.THRESH_BINARY)
    # 将二值化结果与 peephole（窥孔掩模）进行元素级乘法运算，可能用于优化条形码识别区域
    thr = thr * peephole
    # 判断二值化图像是否有足够的有效像素（面积较小可能是无条形码）
    if thr.sum() < 200:
        # print("EMPTY")
        return ("empty", "empty")

    # # 开始计时
    # start = time.time()
    # 尝试直接解码原始图像
    code = decode_raw(well)
    # # 计算识别耗时
    # duration = time.time() - start
    # print(f"Elapsed time decode_raw: {duration*1000} ms")
    if code:
        # print(f"decode_raw: {code}")
        return (code, "raw")

    # （已注释掉的LSD解码尝试）
    # code = decode_lsd(well)
    # if code:
    #     return (code, 'lsd')

    # # 开始计时
    # start = time.time()
    # 使用Harris角点检测增强后解码
    code = decode_harris(well)
    # # 计算识别耗时
    # duration = time.time() - start
    # print(f"Elapsed time decode_harris: {duration*1000} ms")
    if code:
        # print(f"decode_harris: {code}")
        return (code, "harris")

    # # 开始计时
    # start = time.time()
    # 在未经改进的原始图像上尝试解码
    code = decode(well)
    # # 计算识别耗时
    # duration = time.time() - start
    # print(f"Elapsed time decode_harris: {duration*1000} ms")
    if code:
        # print(f"decode_unchanged: {code}")
        return (code, "unchanged")

    # 通过傅里叶变换进行图像旋转校正后尝试解码
    rotated = improve_fft(well)
    code = decode(rotated)
    if code:
        return (code, "rotated")

    # 所有尝试解码均失败，返回失败标记
    # print("FAILED")
    return ("failed", "failed")


def decode_lsd(well, debug=False):
    """
    decode_lsd 函数使用 LSD（Line Segment Detector）算法检测输入井图像（well）中的线段，并根据最长的两条线段构建一个矩形框，然后对图像进行仿射变换和解码操作。

    Args:
        well (_type_): 输入的二维数组或灰度图像数据。
        debug (bool, optional): 是否开启调试模式以显示中间过程和结果。默认为 False。

    Returns:
        _type_: 返回解码得到的条形码字符串，若未成功则返回 None。

    实现流程：
    1. 创建并应用 Line Segment Detector 对图像 `well` 进行线段检测，获取所有线段及其宽度、精度和 NFA 值。
    2. 计算所有线段长度，并按长度排序。
    3. 获取最长的两条线段并计算它们的参数表示。
    4. 找到这两条线段的交点。
    5. 根据交点以及两条线段定义一个新的矩形框。
    6. 将矩形框调整至标准顺序（左上、右上、右下、左下）。
    7. 使用这个矩形框对原始图像进行仿射变换及解码操作。

    在调试模式下，函数会绘制包含线段、矩形框和交点的图像，以及可能存在的二值化图像结果。
    """

    # 创建并初始化 Line Segment Detector
    lsd = cv2.createLineSegmentDetector()

    # 检测图像中的线段
    lines, width, prec, nfa = lsd.detect(well)

    # 计算每条线段的长度
    lengths = np.sqrt(
        (lines[:, 0, 0] - lines[:, 0, 2]) ** 2 + (lines[:, 0, 1] - lines[:, 0, 3]) ** 2
    )

    # 按长度升序排列索引
    len_idx = lenghts.argsort()

    # 获取最长的两条线段
    line1 = lines[len_idx[-1]][0]
    line2 = lines[len_idx[-2]][0]

    # 计算两条线段的参数表示
    L1 = line_params(line1)
    L2 = line_params(line2)

    # 计算两条线段的交点
    its = intersection(L1, L2)

    # 构建矩形框顶点 A、B、C、D
    A = dist_point(line1, its)
    B = its
    C = dist_point(line2, its)
    D = A - B + C

    # 定义并调整矩形框顺序
    box = np.array([A, B, C, D])
    box = box[[box.argmax(0)[1], box.argmin(0)[0], box.argmin(0)[1], box.argmax(0)[0]]]

    # 对图像进行仿射变换和解码
    code, binarized = warp(well, box, True)

    # 调试模式下可视化处理结果
    if debug:
        contours_img = cv2.cvtColor(
            well, cv2.COLOR_GRAY2RGB
        )  # 将灰度图转换为RGB以便绘制

        # 绘制矩形框
        polyline = [box.astype(np.int32).reshape(-1, 1, 2)]
        cv2.polylines(contours_img, polyline, True, (0, 255, 255))

        # 绘制最长的两条线段
        lsd.drawSegments(contours_img, lines[len_idx[-2:]])

        # 绘制两条线段交点
        cv2.circle(contours_img, its, 5, (255, 0, 0))

        # 显示带有线段、矩形框和交点的图像
        plt.subplot(132)
        plt.imshow(contours_img)

        # 显示二值化后的图像
        plt.subplot(133)
        plt.imshow(binarized)

        plt.show()

    return code


def decode_raw(well, debug=False):
    """
    decode_raw 函数对输入的孔位图像（well）进行预处理，识别条形码区域，并根据不同的条件调用 warp 函数进行解码。

    Args:
        well (_type_): 输入的二维数组或灰度图像数据。
        debug (bool, optional): 是否开启调试模式以显示中间过程和结果。默认为 False。

    Returns:
        _type_: 返回解码得到的条形码字符串，若未成功则返回 None。

    实现流程：
    1. 对输入图像应用阈值并找到轮廓。
    2. 计算并拟合一个最小外接矩形框（box），获取其长宽（a、b）。
    3. 根据矩形框尺寸判断是否满足特定范围（65-80像素之间），如果满足，则修剪轮廓并对图像进行仿射变换及解码操作。
    4. 若矩形框尺寸接近另一组条件（宽度接近50像素且高度接近94像素），在轮廓上添加额外点后重新拟合矩形框并解码。
    5. 其他情况下，直接返回 None 和全白色的二值化图像作为失败结果。

    在调试模式下，函数将绘制原始图像、带有轮廓和矩形框标注的图像以及可能存在的二值化图像结果。
    """

    # 找到图像中的轮廓并计算阈值后的轮廓
    cntr = find_contour(threshold(well))

    # 计算并拟合一个最小外接矩形框
    box, u, v, a, b = fit_box(cntr)
    center_b = box[0] + (box[2] - box[0]) / 2
    # rect = cv2.minAreaRect(cntr)
    # box = cv2.boxPoints(rect)
    # center_b, size, _ = rect
    # a, b = size

    # 排序矩形框的长宽
    a, b = sorted([a, b])
    # if a > b:
    #     a, b = b, a

    # 判断矩形框尺寸是否满足特定范围（65至80像素）
    if a > 65 and a < 80 and b > 65 and b < 80:
        # 裁剪轮廓并使用warped函数进行解码
        trimmed_cntr = trim_contour(cntr)
        code, binarized = warp(well, trimmed_cntr, debug=True)
    # 判断矩形框尺寸是否接近特定条件（宽度接近50像素且高度接近94像素）
    elif abs(a - 50) < 5 and abs(b - 94) < 5:
        # 计算矩形框中心点并找到最小包围圆心
        c, r = cv2.minEnclosingCircle(cntr)
        center_c = np.array(c)

        # 添加一个基于圆心的额外点到轮廓中
        extra_point = np.int32([[2 * center_c - center_b]])
        cntr = np.append(cntr, extra_point, axis=0)

        # 重新拟合矩形框并进行解码
        # rect = cv2.minAreaRect(cntr)
        # box = cv2.boxPoints(rect)
        box, u, v, a, b = fit_box(cntr)
        code, binarized = warp(well, box, debug=True)
    # 其他情况，返回None作为解码结果
    # elif a > 80 or b > 80:
    else:
        code, binarized = None, np.ones_like(well)  # 创建全白的二值图像

    # 调试模式下绘制并显示图像
    if debug:
        plt.subplot(131)
        plt.title("well")
        plt.axis("off")
        plt.imshow(well)

        raw_rgb = cv2.cvtColor(well, cv2.COLOR_GRAY2RGB)  # 将灰度图转换为RGB以便绘制
        cv2.drawContours(raw_rgb, [np.int0(box)], 0, (255, 0, 0), 1)  # 绘制矩形框
        cv2.drawContours(raw_rgb, cntr, -1, (0, 0, 255), 1)  # 绘制所有轮廓

        plt.subplot(132)
        plt.title("contours")
        plt.axis("off")
        plt.imshow(raw_rgb)

        # 如果“binarized”变量存在，则显示它
        if "binarized" in vars():
            plt.subplot(133)
            plt.title("binarized")
            plt.axis("off")
            plt.imshow(binarized)

        plt.show()

    return code


def decode_harris(well, debug=False, harris=None):
    """
    decode_harris 函数通过使用 Harris 角点检测算法对输入的井图像（well）进行处理，识别可能包含条形码区域的矩形框，并调用 warp 函数进行进一步解码。

    Args:
        well (_type_): 输入的二维数组或灰度图像数据。
        debug (bool, optional): 是否开启调试模式以显示中间过程和结果。默认为 False。
        harris (_type_, optional): 预计算的 Harris 角点响应图。如果没有提供，则函数内部会自动计算。默认为 None。

    Returns:
        _type_: 返回解码得到的条形码字符串，若未成功则返回 None。

    实现流程：
    1. 如果 `harris` 参数为空，计算输入图像的 Harris 角点响应图。
    2. 根据 Harris 响应图计算偏斜度，如果偏斜度过大（表明元素接近正方形），应用形态学闭运算对其进行平滑处理。
    3. 对 Harris 响应图应用阈值并寻找轮廓。
    4. 计算一个拟合矩形框（box），判断其长宽是否大于预设的最小尺寸。
    5. 若满足条件，则基于找到的矩形框进行仿射变换及后续处理，尝试解码条形码；否则不进行处理直接返回原始图像。
    6. 如果调试模式开启，绘制并显示多个中间步骤的图像结果，包括原始图像上的轮廓、经过处理的warped图像以及binarized二值化图像。

    注意：在实际代码中，min_size 应该是定义好的变量，表示要保留的最小矩形框尺寸。
    """

    # 计算Harris角点响应图，如果参数harris没有提供
    harris = cv2.cornerHarris(well, 4, 1, 0.0)

    # 计算响应图的偏斜度，根据偏斜度调整形状
    skew = scipy.stats.skew(harris, axis=None)
    if skew > 3.49:  # 检查是否近似正方形
        harris = cv2.morphologyEx(harris, cv2.MORPH_CLOSE, make_round_kernel(9))

    # 应用阈值处理
    thr = threshold(harris, 0.1)

    # 寻找图像中的轮廓
    cntr = find_contour(thr)

    # 计算并拟合一个最小外接矩形框
    # rect = cv2.minAreaRect(cntr)
    # box = cv2.boxPoints(rect)
    # _, size, _ = rect
    # a, b = size
    box, u, v, a, b = fit_box(cntr)

    # 判断矩形框尺寸是否满足最小要求
    if a > min_size and b > min_size:
        # 对矩形框进行修剪并执行仿射变换等操作
        box = trim_contour(cntr)
        code, binarized = warp(well, box, debug=True, thr_level=80)
    else:
        # 不满足条件时，返回空结果
        code, binarized = None, well

    # 调试模式下绘制和显示图像
    if debug:
        # 将灰度图像转换为RGB以便绘制
        contours_rgb = cv2.cvtColor(well, cv2.COLOR_GRAY2RGB)

        # 在图像上绘制所有轮廓和拟合后的矩形框
        contours_rgb = cv2.drawContours(contours_rgb, [np.int0(box)], 0, (255, 0, 0), 1)
        contours_rgb = cv2.drawContours(contours_rgb, cntr, -1, (0, 0, 255), 1)
        contours_rgb = cv2.drawContours(contours_rgb, [np.int0(box)], 0, (255, 0, 0), 1)
        rect = cv2.minAreaRect(cntr)
        orig_box = cv2.boxPoints(rect)
        contours_rgb = cv2.drawContours(
            contours_rgb, [np.int0(orig_box)], 0, (255, 255, 0), 1
        )

        # 显示图像子集
        plt.subplot(132)
        plt.title("contours")
        plt.axis("off")
        plt.imshow(contours_rgb)

        # 如果存在“binarized”变量，则显示它
        if "binarized" in vars():
            plt.subplot(133)
            plt.title("binarized")
            plt.axis("off")
            plt.imshow(binarized)
        plt.show()

    return code


def warp(well, box, debug=False, thr_level=80):
    """
    warp 函数对输入的井图像（well）进行一系列图像处理操作，包括仿射变换、裁剪、调整大小、阈值处理、边界检查和修复，并尝试解码得到条形码信息。

    Args:
        well (_type_): 输入的二维数组，代表原始井图像。
        box (_type_): 四个点构成的数组，表示最小外接矩形的顶点坐标，用于计算仿射变换矩阵。
        debug (bool, optional): 是否开启调试模式以显示中间处理结果。默认为 False。

    Returns:
        _type_: 返回一个包含以下内容的元组：
            - code: 解码成功的条形码字符串，若未成功则为 None。
            - binarized: 经过处理后的二值化井图像数据。

    实现流程：
    1. 根据关键字参数获取阈值级别（thr_level，默认为80）。
    2. 计算仿射变换矩阵 M，将井图像的部分区域（根据box确定）拉伸至大小为120x120的正方形。
    3. 应用仿射变换得到变形后的图像warped。
    4. 遍历预设的尺寸列表dm_size，对变形后的图像进行缩放并应用阈值处理生成二值图像thr2。
    5. 使用border_check_fix函数检查并可能修复二值图像的边界。
    6. 若边界修复成功，则添加边框并对图像进行进一步调整，以便进行解码操作。
    7. 对经过处理的二值图像进行解码，如果成功得到条形码信息则跳出循环。
    8. 如果debug模式开启，将中间处理结果显示出来；否则直接返回原输入井图像作为binarized。
    """

    # 定义目标尺寸
    a = 120

    # 确定仿射变换的源点坐标
    if box[1, 1] < box[3, 1]:
        src = box[0:3]
    else:
        src = box[1:4]

    # 计算仿射变换矩阵
    M = cv2.getAffineTransform(src, np.array([[0, a], [0, 0], [a, 0]], dtype="float32"))

    # 对井图像进行仿射变换
    warped = cv2.warpAffine(well, M, (a, a))

    # 初始化解码结果
    code = None

    # 遍历预设尺寸列表
    for size in dm_size:
        # 调整图像尺寸
        resized = cv2.resize(warped, (size, size))

        # 应用阈值处理
        thr2 = threshold(resized, thr_level)

        # 边界检查与修复
        if border_check_fix(thr2, size):
            # 添加边框
            barcode = cv2.copyMakeBorder(thr2, 2, 2, 2, 2, cv2.BORDER_CONSTANT)
            barcode = cv2.resize(
                barcode, (80, 80), 0, 0, interpolation=cv2.INTER_NEAREST
            )
            barcode_cl = cv2.cvtColor(barcode, cv2.COLOR_GRAY2BGR)

            # 尝试解码条形码
            code = decode(barcode_cl)
            if code:
                break

    # 如果开启调试模式，生成并返回显示中间过程的图像
    if debug:
        # 将warped转换为三通道图像以便显示
        binarized = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        mask = cv2.resize(thr2, warped.shape, interpolation=cv2.INTER_NEAREST) / 255
        if code:
            binarized[:, :, 1] = warped * mask * 0.7
            binarized[:, :, 0] = warped * (1 - mask)
        else:
            binarized[:, :, 0] = warped * mask * 0.7
            binarized[:, :, 1] = warped * (1 - mask)

        binarized[:, :, 2] = 0
    else:
        # 若非调试模式，返回未经处理的原始井图像
        binarized = well

    return code, binarized


def border_check(arr):
    """
    border_check 函数用于检查输入的二维数组（arr）是否符合 Data Matrix 矩阵的边界特征。

    Args:
        arr: 一个二维布尔型或整数型数组，代表数据矩阵图像。值为0表示背景，非零值表示前景。

    Returns:
        bool: 如果输入数组满足以下条件，则返回 True，否则返回 False：
            - 数据矩阵是正方形，并且边长为偶数。
            - 边界的行/列元素数量满足特定模式（即上下和左右边界分别有一半的元素为1，另一半为0）。
            - 上下两条边界中0与1交替出现的位置间隔为奇数个单位。

    实现流程：
    1. 获取输入数组的大小（假设为正方形）。
    2. 将输入数组转换为布尔型数组，其中值大于0的视为True（前景）。
    3. 提取并计算数组的四条边界（上、右、下、左）中1的个数。
    4. 检查边界1的数量是否符合预期模式：上/下边界的一半数量为1，左/右边界所有数量均为1。
    5. 对边界1的数量进行排序后，检查最接近的两个最大值之间的差是否为奇数。
    6. 根据上述检查结果，返回布尔值表示该数组的边界是否满足 Data Matrix 的要求。
    """

    # 获取数组大小
    size = arr.shape[0]

    # 断言验证：数组应为正方形且边长为偶数
    assert (
        size == arr.shape[1] and size % 2 == 0
    ), "Data Matrix should be square and of even size."

    # 将数组转换为布尔型数组，便于后续处理
    arr = arr > 0

    # 计算四条边界中1的数量
    border = np.array([arr[-1, :], arr[:, 0], arr[0, :], arr[:, -1]]).sum(1)

    # 检查边界中1的数量分布是否符合 Data Matrix 要求
    if not np.array_equal(np.sort(border), [size / 2, size / 2, size, size]):
        return False

    # 对边界中1的数量排序，并找到最接近的最大值索引
    b_index = border.argsort()

    # 检查上下边界中1的间距是否为奇数个单位
    if abs(b_index[-1] - b_index[-2]) % 2 != 1:
        return False

    # 若所有检查均通过，则返回 True
    return True


def border_check_fix(arr, size):
    """
    border_check_fix 函数用于检查一个二维数组（arr）的边界元素是否符合特定模板，并在不满足时尝试修复它。

    Args:
        arr: 一个二维整数数组，代表图像或矩阵。
        size: 整数值，表示模板大小。目前支持12和14两种尺寸。

    Returns:
        返回一个布尔值，表示操作结果：
            - 如果原始边界的元素经过检查和修复后满足给定模板，则返回 True；
            - 如果无法修复或者原始边界与任何模板都不匹配，则返回 False。

    实现流程：
    1. 提取二维数组 arr 的四条边界并存储到 borders 数组中。
    2. 将边界上非零元素设置为1，便于后续计算。
    3. 根据 size 参数选择相应的模板（一种特殊的边界模式）。
    4. 计算当前边界与模板之间的差异，并将差异累加。
    5. 检查整个边界上的元素数量是否与预期一致，如果不一致，则直接返回 False。
    6. 检查最小差异对应的模板，如果其差异大于4，则说明不符合任何模板，返回 False。
    7. 如果差异为0，说明边界完全符合某个模板，返回 True。
    8. 若边界不完全符合但可以修复，使用最小差异模板替换原边界，并将替换后的边界元素恢复到原始的最大值，最后更新 arr 的边界元素。
    9. 在所有操作完成后，返回 True，表示已成功修复边界使其符合模板要求。
    """

    # return True

    # 提取二维数组的边界
    borders = np.array([arr[-1, :], arr[:, 0], arr[0, :], arr[:, -1]])

    # 缩小边界元素以便进行求和运算
    borders[borders > 0] = 1

    # 根据 size 参数选择合适的模板
    if size == 12:
        template = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            ]
        )
    elif size == 14:
        template = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            ]
        )

    # 计算边界与模板之间的差异，并求和
    diffs = np.array([np.logical_xor(borders, x).sum(1) for x in template])

    # 获取最小差异值之和
    wrong = diffs.min(0).sum()

    # 检查边界元素总数是否与预期相符
    if abs(borders.sum() - 3 * size) > 4:
        return False

    # 检查与模板的差异是否可接受
    elif wrong > 4:
        return False
    elif wrong == 0:
        return True

    # 边界需要修复且可以修复的情况
    else:
        # 使用最匹配的模板替换原边界
        borders = template[diffs.argmin(0)]

        # 将替换后的边界元素恢复到原始的最大值
        borders[borders > 0] = arr.max()

        # 更新二维数组 arr 的边界元素
        arr[-1, :] = borders[0]
        arr[:, 0] = borders[1]
        arr[0, :] = borders[2]
        arr[:, -1] = borders[3]

        # 表示成功修复边界，返回 True
        return True


def find_contour(img):
    """
    find_contour 函数用于在输入图像中查找面积最大的外部轮廓，并检查其尺寸是否符合特定范围。

    Args:
        img (_type_): 输入的二维数组，表示一个灰度图像。如果该图像不是 uint8 类型，则会先将其转换为 uint8 类型以满足 OpenCV 轮廓检测的要求。

    Returns:
        _type_: 返回找到的第一个满足条件（最小坐标大于1且最大坐标小于well_size - 2）的轮廓。如果没有找到符合条件的轮廓，则返回 None 或默认值 np.array([[148,1],[1,1],[1,148],[148,148]], dtype=np.float32)。

    函数流程：
    1. 检查输入图像的数据类型，如果不是 uint8 类型，则转换为 uint8 类型。
    2. 使用 OpenCV 的 `cv2.findContours` 函数查找图像中的所有外部轮廓，并获取轮廓及其层次结构信息。
    3. 按照轮廓面积从大到小对轮廓进行排序。
    4. 遍历排序后的轮廓列表，对于每一个轮廓，检查其最小和最大坐标是否在指定范围内。
       - 如果满足条件，则直接返回当前轮廓。
       - 如果遍历结束仍未找到满足条件的轮廓，则返回最后一个轮廓或预设的默认值。
    """

    if img.dtype != "uint8":
        img = img.astype("uint8")

    # 找到图像中的所有外部轮廓
    cntrs, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 按照轮廓面积大小降序排列
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)

    # 遍历轮廓列表，寻找第一个满足尺寸条件的轮廓
    for cntr in cntrs:
        if cntr.min() > 1 and cntr.max() < well_size - 2:
            return cntr

    # 如果没有找到满足条件的轮廓，返回 None 或预设的默认轮廓点集
    return cntr  # 这里应该返回 None，代码可能有误；或使用：return None 或者 np.array([...], dtype=np.float32)


def improve_fft(well):
    """
    该函数用于改进二维傅里叶变换（FFT），通过应用一系列图像处理技术，如频率域滤波、高斯模糊、旋转校正等，以优化输入井的图像质量。

    Args:
        well (_type_): 输入变量well是一个二维数组（例如：numpy数组），代表原始井的图像数据。

    Returns:
        _type_: 返回一个经过处理后的二维数组，表示优化和旋转校正后的井图像。

    实现步骤：
    1. 使用numpy库计算well的二维傅里叶变换，并进行fftshift操作，使低频分量位于图像中心。
    2. 创建一个与well相同大小的零掩模矩阵mask。
    3. 计算图像的中心坐标。
    4. 在掩模矩阵中绘制一个内外半径分别为60和50的圆环区域，内部圆以外的部分设为1，内部圆内设为0。这实现了对特定频率成分的筛选。
    5. 将筛选后的掩模应用于傅里叶变换结果filtered。
    6. 对筛选后的结果进行高斯模糊处理以降低高频噪声。
    7. 找到高斯模糊后的图像中的最大值及其坐标。
    8. 根据最大值坐标计算其与图像中心的连线角度theta，可能会遇到除以零的异常情况，此时将theta设为0。
    9. 根据得到的角度theta以及图像中心点，利用cv2.getRotationMatrix2D创建一个旋转矩阵M。
    10. 使用cv2.warpAffine对原始井图像well进行旋转校正，返回旋转后的新图像rotated。
    """
    # 计算二维傅里叶变换并移动低频成分至中心
    fft = np.fft.fftshift(np.fft.fft2(well))

    # 创建掩模矩阵
    mask = np.zeros(well.shape)

    # 计算图像中心坐标
    center = tuple(x // 2 for x in well.shape)

    # 绘制圆环区域作为频域滤波器
    cv2.circle(mask, center, 60, color=1, thickness=-1)
    cv2.circle(mask, center, 50, color=0, thickness=-1)

    # 应用掩模到傅里叶变换结果
    filtered = np.copy(fft) * mask

    # 高斯模糊处理
    blur = cv2.GaussianBlur(np.abs(filtered), (5, 5), 0)

    # 找到最大值的位置
    maximum = cv2.minMaxLoc(blur)[3]

    # 计算角度theta
    try:
        theta = math.atan(float(maximum[1] - center[1]) / float(maximum[0] - center[0]))
    except ZeroDivisionError:
        theta = 0

    # 创建旋转矩阵
    M = cv2.getRotationMatrix2D(center, np.rad2deg(theta), 1.0)

    # 对原始井图像进行旋转校正
    rotated = cv2.warpAffine(well, M, (well.shape[1], well.shape[0]))

    return rotated


def decode(img):
    """
    decode 函数用于解码输入图像（img）中的二维条形码（Data Matrix）并验证其内容格式。

    Args:
        img (_type_): 一个二维数组，通常为灰度图像或单通道彩色图像，表示包含 Data Matrix 条形码的图像数据。

    Returns:
        _type_: 如果成功解码并验证了条形码内容，则返回解码后的字符串；否则返回 None。

    函数实现：
    1. 获取输入图像的高度（height）和宽度（width）。
    2. 使用 pylibdmtx 库中的 decode 函数尝试从图像中解码 Data Matrix 条形码。`**libdmtx_params` 表示可能存在的额外参数（未在代码片段中显示）。
    3. 若解码成功，取出第一个条形码数据，并将其转换为 UTF-8 编码的字符串。
    4. 验证解码得到的字符串是否符合特定格式：可选两个字母开头（`\w\w`），后面跟随10个数字（`\d{10}`）。
    5. 如果解码结果满足预设格式要求，则返回该字符串；否则返回 None。
    """

    # # 获取图像的高和宽
    # height = img.shape[0]
    # width = img.shape[1]

    # 使用pylibdmtx库解码图像中的Data Matrix条形码
    code = pylibdmtx.decode(img, **libdmtx_params)

    # 检查是否有解码结果，并将解码后的字节数据转为UTF-8编码的字符串
    code = code[0].data.decode("utf-8") if code else False

    # 验证解码得到的字符串格式是否符合预设规则
    # if code and re.match("(\w\w)?\d{10}", code):
    return code
    # else:
    #     return None


def trim_contour(cntr, size=70):
    """
    trim_contour 函数用于调整轮廓 (contour) 的形状，使其在指定尺寸范围内尽可能接近正方形。

    Args:
        cntr (_type_): 一个 numpy 数组表示轮廓点集，通常具有 shape (N, 1, 2)，其中 N 是轮廓点的数量。
        size (int, optional): 目标边长范围的上限，默认值为 70。当轮廓某一边长大于此值时，函数会尝试将这一侧的轮廓拉伸或压缩以接近给定的大小限制。

    Returns:
        _type_: 返回经过调整后的最小面积包围矩形 (Minimum Area Bounding Rectangle) 的四个顶点坐标组成的数组。

    函数实现：
    1. 使用 fit_box 函数获取轮廓的最小面积包围矩形及其矩形边长度和方向向量。
    2. 当任一矩形边长度大于设定的 `size` 时，计算对应的缩放矩阵 Mb。
    3. 对轮廓点进行变换，使得过长的矩形边一侧行坐标（根据 dim 确定）被拉伸或压缩至接近目标边长。
    4. 调整轮廓中最大和最小行坐标的点到包围矩形中心，以此来裁剪超出范围的部分。
    5. 在所有矩形边长度均小于等于 `size` 后，返回最终得到的最小面积包围矩形的四个顶点坐标。
    """

    # 循环处理直到轮廓的最小面积包围矩形的两边都不超过给定尺寸
    while True:
        # 计算并获取轮廓的最小面积包围矩形以及相关参数
        # rect = cv2.minAreaRect(cntr)
        # box = cv2.boxPoints(rect)
        # center, rect_size, _ = rect
        # a, b = rect_size
        # u = box[0] - box[1]
        # v = box[2] - box[1]
        box, u, v, a, b = fit_box(cntr)

        center = box[0] + (box[2] - box[0]) / 2

        # break

        # 根据对角线较长的一侧决定处理维度（dim）
        if a > size:
            dim = 0
        elif b > size:
            dim = 1
        else:
            # 如果两条对角线都满足条件，则退出循环
            break

        # 计算归一化的对角线方向向量
        u1, v1 = u / a, v / b

        # 创建缩放矩阵 Mb
        Mb = np.column_stack((u1, v1))

        # 将轮廓点转换为列向量，并进行矩阵乘法变换
        cntp = cntr.copy()[:, 0, :].T
        cntp = np.dot(Mb.T, cntp)

        # 获取变换后轮廓在当前处理维度上的坐标
        arr = cntp[dim, :]

        # 计算该维度上轮廓坐标的平均值
        m = arr.mean()

        # 找出该维度上坐标的最大值和最小值索引
        imax, imin = arr.argmax(), arr.argmin()

        # 将最大值和最小值更新为平均值，实现拉伸或压缩效果
        cntp[dim, imax] = m
        cntp[dim, imin] = m

        # 更新轮廓中最大和最小值对应点的位置到包围矩形的中心
        cntr[imax, 0] = center
        cntr[imin, 0] = center

    # 当退出循环后，返回调整后包围矩形的四个顶点坐标
    return box


def threshold(img, level=0):
    """
    threshold 函数用于对输入图像进行阈值处理，以二值化图像。根据传入的参数 `level` 自动选择合适的阈值方法（全局阈值、自适应阈值或指定阈值）。

    Args:
        img (_type_): 输入图像数据，通常为灰度图像（单通道），表示为一个 numpy 数组。
        level (_type_, optional): 阈值参数。默认为 None。可以是以下几种情况：
            - 如果未提供（或设置为0）且不使用 Otsu 法，则函数会自动使用 Otsu's 二值化方法确定阈值。
            - 如果 level 是一个小于1的浮点数，将其视为相对于图像最大像素值的比例，并以此比例计算实际阈值进行二值化。
            - 如果 level 是一个整数，则直接将这个值作为阈值进行二值化处理。

    Returns:
        _type_: 返回经过阈值处理后的二值图像，是一个与输入图像同样大小的 numpy 数组，其中像素值为0或255，代表背景和前景。

    函数实现：
    根据 level 的类型和值来选择 OpenCV 中的阈值类型 (tt) 并计算实际使用的阈值。
    使用 cv2.threshold 函数对输入图像应用相应的阈值处理方式，返回二值化后的图像。
    """

    # 判断 level 参数是否为 None 或等于 0，如果是则使用 Otsu's 二值化方法
    if level == 0:
        tt = cv2.THRESH_OTSU
        level = 0

    # 若 level 是一个小于1的浮点数，按其占图像最大像素值的比例计算阈值
    elif 0 < level < 1:
        tt = cv2.THRESH_BINARY

        # level = level * img.max()
        _, max_val, _, _ = cv2.minMaxLoc(img)
        level *= max_val

    # 若 level 是一个整数，直接设定为阈值
    else:
        tt = cv2.THRESH_BINARY

    # 应用阈值处理
    level, thr = cv2.threshold(img, level, 255, tt)

    # 返回二值化后的图像
    return thr


def fit_box(cntr):
    """
    fit_box 函数用于找到一个最小面积包围矩形 (Minimum Area Bounding Rectangle, MABR) 并计算相关参数。

    Args:
        cntr (_type_): 通常为一个 numpy 数组，表示一组二维点（可能是轮廓点或其他形状的顶点），格式为 (N, 2)，其中 N 是点的数量，每一行代表一个点的 (x, y) 坐标。

    Returns:
        _type_: 返回以下多个对象：
            - box: 一个包含矩形四个顶点坐标的数组，每个顶点是一个二元组 `(x, y)`，顺序是逆时针方向从左上角开始。
            - u: 矩形对角线之一的向量，计算方法是从矩形的一个顶点减去另一个顶点。
            - v: 矩形另一条对角线的向量，同样通过两个顶点相减得到。
            - norm_u: 向量 `u` 的范数（长度）。
            - norm_v: 向量 `v` 的范数（长度）。

    函数实现：
    1. 使用 OpenCV 的 cv2.minAreaRect 函数找到输入点集 `cntr` 的最小面积外接矩形，并返回其旋转框（Rotated Rectangle）表示。
    2. 调用 cv2.boxPoints 函数将旋转框转换为边界框的四个角点坐标。
    3. 计算矩形两条对角线的向量 `u` 和 `v`，这里取的是相对顶点进行计算。
    4. 使用 OpenCV 的 cv2.norm 函数分别计算这两条对角线的长度（范数）并返回。
    """

    # 寻找给定点集 cntr 的最小面积外接矩形，并获取其四个顶点坐标
    box = cv2.boxPoints(cv2.minAreaRect(cntr))

    # 计算矩形第一条对角线的向量
    u = box[0] - box[1]

    # 计算矩形第二条对角线的向量
    v = box[2] - box[1]

    # 计算两对角线的长度（范数）
    norm_u = cv2.norm(u)
    norm_v = cv2.norm(v)

    # 返回矩形的四个顶点、两条对角线向量及其长度
    return box, u, v, norm_u, norm_v


def make_round_kernel(size):
    """
    make_round_kernel 函数用于生成一个圆形的二值内核（通常用于图像处理，如滤波、形态学操作等）。

    Args:
        size (_type_): 一个整数，表示要生成的圆形内核的大小，它应该是奇数以确保中心像素存在。

    Returns:
        _type_: 返回一个 numpy 数组，形状为 (size, size)，数据类型为 np.uint8。该数组代表了一个圆形的二值内核，其中值为1的部分构成了内核的圆形区域，其余部分值为0。

    函数实现：
    1. 首先创建一个指定大小的零矩阵作为初始内核。
    2. 计算圆心位置，它位于内核的中心，即行和列索引均为半径。
    3. 使用 OpenCV 的 cv2.circle 函数在内核中心绘制一个填充的圆形，颜色设置为1（在二值图像中，1代表前景），厚度设为-1表示填充整个圆。
    4. 返回生成的圆形内核。
    """

    # 创建一个指定大小且所有元素都为0的二维numpy数组
    kernel = np.zeros((size, size), np.uint8)

    # 计算圆形内核的半径，由于是正方形内核，所以半径就是边长的一半
    r = size // 2

    # 在内核中心绘制一个填充的圆形，颜色值为1
    kernel = cv2.circle(kernel, (r, r), r, color=1, thickness=-1)

    # 返回构建好的圆形内核
    return kernel


def dist(p1, p2):
    """
    dist 函数用于计算两个二维空间点之间的欧几里得距离。

    Args:
        p1 (_type_): 一个包含两个元素（通常是浮点数）的数组或列表，表示第一个点的坐标 (x1, y1)。
        p2 (_type_): 一个包含两个元素（同样是浮点数）的数组或列表，表示第二个点的坐标 (x2, y2)。

    Returns:
        float: 返回一个浮点数，表示从点 p1 到点 p2 的直线距离。

    函数实现：
    欧几里得距离是两点在直角坐标系中坐标的平方差之和的平方根。这个函数通过使用勾股定理来计算两点间的这种距离。
    """

    # 计算在 x 轴和 y 轴上的坐标差值的平方
    dx_squared = (p1[0] - p2[0]) ** 2
    dy_squared = (p1[1] - p2[1]) ** 2

    # 将平方差相加并取平方根以得到两点间的欧几里得距离
    return math.sqrt(dx_squared + dy_squared)


def dist_point(line, B):
    """
    dist_point 函数用于计算点 B 到直线（由参数 line 表示）的垂直投影点，并返回从点 B 到该投影点的距离。

    Args:
        line (_type_): 一个包含四个元素的数组或列表，表示直线。line[:2] 和 line[2:] 分别为直线上的两个点坐标，用于确定直线方向。
        B (_type_): 一个包含两个元素的数组或列表，表示需要计算距离的点的坐标。

    Returns:
        _type_: 返回一个包含两个元素的数组或列表，表示点 B 到其在直线上的垂直投影点的距离向量。

    函数执行以下步骤：
    1. 根据点 B 与直线上的两个点之间的距离，选择离 B 较近的一个点作为参照点 A。
    2. 计算点 B 与参照点 A 的向量差 v。
    3. 将向量 v 正规化并乘以固定长度值68，生成单位向量沿直线法线方向的向量。
    4. 从点 B 减去这个单位法向量得到的结果即为从点 B 到其在直线上垂直投影点的距离向量。

    注意：这里假设 cv2.norm 是计算向量范数的函数，68是一个预设的常数值，用于调整垂直于直线的距离尺度。
    """

    # 确定距离点 B 较近的直线上的参照点 A
    A = line[:2] if dist(line[:2], B) > dist(line[2:], B) else line[2:]

    # 计算从参照点 A 到点 B 的向量差 v
    v = B - A

    # 正规化向量 v 并将其长度调整为68，从而得到单位法向量
    v = v / cv2.norm(v) * 68

    # 计算并返回从点 B 到其垂直投影点的距离向量
    return B - v


def line_params(pp):
    """
    line_params 函数用于从给定的两个点 pp 计算直线的一般式参数 (A, B, C)。

    Args:
        pp (_type_): 一个至少包含四个元素的一维或二维数组。当为一维时，pp[0:4] 应该是表示直线上的两点坐标，格式为 [x1, y1, x2, y2]；当为二维时，假设输入为多组点，函数将只处理第一组点。

    Returns:
        _type_: 返回一个包含三个元素的元组 (A, B, C)，代表直线的一般式 Ax + By + C = 0 中的系数 A、B 和 C。

    此函数首先检查输入数据维度，如果大于一维，则选取第一个点对作为计算对象。然后根据两点坐标计算直线的方向向量（A, B）以及常数项 C，从而得到直线的一般式参数。
    """

    # 如果输入是一个多维数组，取其第一个点对进行直线参数计算
    if pp.ndim > 1:
        pp = pp[0]

    # 根据两点坐标计算直线方向向量的分量
    A = pp[1] - pp[3]
    B = pp[2] - pp[0]

    # 根据两点坐标计算一般式中的常数项 C
    C = pp[0] * pp[3] - pp[2] * pp[1]

    # 返回直线的一般式参数 (A, B, -C)
    return A, B, -C


def intersection(L1, L2):
    """
    intersection 函数用于计算两条直线（在笛卡尔坐标系中，由线方向向量和一个点表示）的交点。

    Args:
        L1 (_type_): 一个包含三个元素的列表或元组，表示第一条直线。其中，L1[0] 和 L1[1] 分别代表直线的方向向量的第一个和第二个分量（斜率），L1[2] 是直线上的一个已知点的横坐标。
        L2 (_type_): 同上，表示第二条直线的参数。

    Returns:
        _type_: 如果两条直线相交，则返回一个包含两个元素的元组，表示交点的坐标 (x, y)；若两条直线平行（即不相交），则返回 None。

    此函数通过计算两条直线的系数得出交点的 x 和 y 坐标：
    1. 计算两直线方向向量的外积 D，用于判断是否相交（D 非零表示两直线不平行，即相交）。
    2. 若 D 非零，则计算交点的横坐标 x 和纵坐标 y，并返回交点坐标 (x, y)。
    3. 若 D 等于零，则说明两条直线平行，无交点，返回 None。
    """

    # 计算两直线方向向量的外积
    D = L1[0] * L2[1] - L1[1] * L2[0]

    # 根据直线方程计算可能的交点横坐标和纵坐标
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]

    # 判断并返回交点坐标（如果存在）
    if D != 0:
        # 计算交点坐标
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        # 当两直线平行时返回 None
        return None
