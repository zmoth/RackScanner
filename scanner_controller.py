""" https://github.com/denisenkom/pytwain """

import platform

if platform.system() == "Windows":
    import twain


class TwainutlError(Exception):
    pass


scanner_defaults = {
    "pixeltype": 2,  # 0: BW, 1: grayscale, 2: RGB
    "resolution": 600.0,
    "autocrop": 0,
    "left": 0,
    "top": 0,
    "right": 3.5,
    "bottom": 5,
}

# override defaults here
scanners = {"AVA6": {}}


def scan(fullfilename="dg_pic.bmp", **kwargs):
    """
    scan 函数用于控制扫描仪进行一次完整的图像扫描操作，并将扫描结果保存为指定文件名的BMP格式图片。

    参数:
        fullfilename (str, optional): 扫描结果输出的完整文件路径，默认为 "dg_pic.bmp"。这个参数指定了存储扫描图像的文件名和路径。

    **kwargs: 可变关键字参数，允许用户自定义扫描属性，例如分辨率、像素类型等。

    步骤说明：
    1. 调用 `open_scanner` 函数获取与TWAIN接口连接的SourceManager对象（sm）以及已打开并准备就绪的Scanner对象（scanner）。
    2. 使用 `adjust_scanner_properties` 函数根据传入的自定义参数(**kwargs)调整扫描器的各项属性设置。
    3. 发送请求开始执行扫描操作，参数(0, 0)表示不显示扫描仪用户界面直接开始扫描（如果需要显示界面，则使用(1, 1)）。
    4. 获取扫描得到的原始图像数据及其是否还有更多图像待处理的信息。
    5. 将原始DIB图像数据转换为BMP格式，并保存到指定的fullfilename中。
    6. 释放扫描得到的原始图像数据占用的内存资源。

    最后确保在函数结束时关闭并释放所有与扫描相关的资源，即使出现异常也会执行这一操作：
    - 关闭并销毁与Scanner关联的对象。
    - 关闭并销毁SourceManager对象。
    """
    sm, scanner = open_scanner()
    try:
        adjust_scanner_properties(scanner, **kwargs)
        scanner.RequestAcquire(0, 0)  # 若需显示扫描仪UI，可改为(1, 1)
        handle, more_to_come = scanner.XferImageNatively()
        twain.DIBToBMFile(handle, fullfilename)
        twain.GlobalHandleFree(handle)
    finally:
        scanner.destroy()
        sm.destroy()


def adjust_scanner_properties(scanner, **kwargs):
    """
    adjust_scanner_properties 函数用于调整扫描仪的特定属性，使其符合预设或自定义参数。

    参数:
        scanner (twain.Source): 已经通过TWAIN接口打开并准备就绪的扫描设备对象。
        **kwargs: 可变关键字参数，允许用户传递自定义扫描参数以覆盖默认设置。

    此函数首先创建一个参数字典（params），该字典包含了所有扫描仪的默认属性值，
    然后用传入的自定义参数（kwargs）更新这些默认值。

    接下来获取当前连接扫描仪的类型名称，并检查是否在预置的 `scanners` 字典中有针对此类型的具体配置，
    如果有，则进一步应用这些特定配置到参数字典中。

    然后，计算出扫描区域的边界框（frame），它由左、上、右、下四个坐标值组成，
    并调用 `scanner.SetImageLayout` 方法来设置扫描图像的布局。

    随后，根据参数字典中的设置，分别调用 `scanner.SetCapability` 方法对扫描仪进行以下属性配置：
    - 设置像素类型（ICAP_PIXELTYPE）
    - 设置分辨率（ICAP_YRESOLUTION）
    - 关闭自动边框检测功能，即实现自动裁剪效果（ICAP_AUTOMATICBORDERDETECTION）

    注释掉的部分提到了可能需要但此处未使用的其他扫描特性配置选项：
    - 'deskew': 使用 ICAP_AUTOMATICDESKEW 标识符可以控制是否启用自动倾斜校正功能
    - 'barcodes': 使用 ICAP_BARCODEDETECTIONENABLED 标识符可以控制是否开启条形码检测功能
    """

    # 创建一个包含默认扫描参数的字典，并用传入的自定义参数覆盖
    params = scanner_defaults.copy()
    params.update(kwargs)
    # 获取当前扫描仪的类型名称
    scanner_type = scanner.GetSourceName()
    # 若当前扫描仪类型在预设的扫描仪配置字典中存在，则应用该类型的特定配置
    if scanner_type in scanners:
        params.update(scanners[scanner_type])
    # 初始化文档、页码和帧号为1
    DocNumber, PageNumber, FrameNumber = 1, 1, 1
    # 计算扫描区域边界框
    frame = tuple(float(params[key]) for key in ["left", "top", "right", "bottom"])
    # 设置扫描区域布局
    scanner.SetImageLayout(frame, DocNumber, PageNumber, FrameNumber)
    # 设置像素类型
    scanner.SetCapability(
        twain.ICAP_PIXELTYPE, twain.TWTY_UINT16, int(params["pixeltype"])
    )
    # 设置垂直分辨率
    scanner.SetCapability(
        twain.ICAP_YRESOLUTION, twain.TWTY_FIX32, float(params["resolution"])
    )
    # 关闭自动边框检测功能
    scanner.SetCapability(
        twain.ICAP_AUTOMATICBORDERDETECTION, twain.TWTY_BOOL, 0
    )  # autocrop
    # 'deskew': twain.ICAP_AUTOMATICDESKEW,
    # 'barcodes': twain.ICAP_BARCODEDETECTIONENABLED,


def open_scanner():
    """
    open_scanner 函数用于打开并初始化与TWAIN兼容的扫描设备。

    概要：
    此函数通过 TWAIN 协议库，查找并打开可用的扫描源（如扫描仪）。如果找到一个或多个扫描源，则尝试连接至默认或用户选择的扫描仪。

    异常：
    TwainutlError: 如果找不到任何可用的 TWAIN 扫描源，函数将抛出此异常，附带错误信息 "No TWAIN sources available."。

    返回值：
    Tuple[twain.SourceManager, twain.Source]: 函数成功执行后返回一个包含 SourceManager 对象和已打开的 Source 对象的元组。
        - sm (twain.SourceManager): 用于管理TWAIN源的SourceManager对象。
        - scanner (twain.Source): 表示已连接并准备就绪的扫描设备的Source对象。
    """

    # 创建一个SourceManager实例，并获取所有可用的TWAIN源列表
    sm = twain.SourceManager(0)
    sourcenames = sm.GetSourceList()

    # 判断可用的TWAIN源数量
    if len(sourcenames) == 0:
        # 若没有找到可用的TWAIN源，则抛出TwainutlError异常
        raise TwainutlError("No TWAIN sources available.")
    elif len(sourcenames) == 1:
        # 如果只有一个可用源，则直接打开该源
        scanner = sm.OpenSource(sourcenames[0])
    else:
        # 如果存在多个源，则交互式地让用户选择并打开一个源
        scanner = sm.OpenSource()

    # 返回SourceManager对象和已打开的Source对象
    return sm, scanner
