import sys
import os
import datetime
import re
from typing import Optional

import pandas as pd
import scanner_controller
import dm_reader
import matplotlib as mpl
import importlib

importlib.reload(scanner_controller)
importlib.reload(dm_reader)
try:
    import settings

    importlib.reload(settings)
except ImportError:
    import settings_template as settings

os.chdir(os.path.abspath(os.path.dirname(__file__)))

for subdir in ["bmp", "csv"]:
    if not os.path.exists(subdir):
        os.mkdir(subdir)

head_template = """
<!DOCTYPE html>
<html>
<head>
    <title>%(title)s</title>
    <link type="text/css" rel="stylesheet" href="resources/rackscanner.css" />
</head>
<body>
    <img src="resources/logo.png" id="logo"/>
    <h1>%(title)s</h1>
    <p class="hint">Place the tube or rack with its A1 position to the
    upper left corner or the scanning area. </p>
<form name="scan" method="get">
%(barcode)s
<button type="submit" name="action" value="rack">%(verb)s Rack</button>
<button type="submit" name="action" value="vial">%(verb)s Single Tube</button>
"""

foot_template = """
</form>
</body>
</html>
"""


def run(**params):
    """
    run 函数是 RackScanner 3.0 程序的核心执行函数，根据传入的参数字典 `params` 来决定执行的操作。此函数主要负责处理扫描、读取和上传操作，并调用相应的子模块或方法。

    参数:
        **params (dict): 可变参数字典，包含程序运行所需的关键信息，如 "platebarcode"（板条码）、"action"（动作类型）等。

    功能流程：
    1. 从参数中获取 `platebarcode`，即当前待扫描或读取样本的板条码。
    2. 根据预设的 `settings.mode` 输出页面头部模板，标题为 "RackScanner 3.0"，并动态填充动词部分（Scan, Demo 或 Read），以及在“扫描模式”下显示一个用于输入板条码的文本框。

    3. 获取 `action` 参数，判断用户请求的动作类型是 rack（扫描架）、vial（扫描试管）还是 csv（上传CSV文件）。

    - 如果 `action` 是 "rack" 或 "vial":
      a. 若处于演示模式（`settings.mode == "demo"`），则直接使用预定义的示例图像路径作为 `filename`。
      b. 若处于扫描模式（`settings.mode == "scanner"`）：
         - 使用 `create_filename` 函数根据 `action` 和 `platebarcode` 创建实际的扫描文件名。
         - 对于 vial 动作，调用 `scanner_controller.scan` 扫描指定位置（右0.5，底部0.5）。
         - 对于 rack 动作，仅调用 `scanner_controller.scan` 进行全幅扫描。
      c. 其他情况下（例如，读取上次扫描的结果），从指定目录 (`settings.images_dir`) 中加载上一次的图片文件。

      在以上任一情况下，都会对扫描得到的图像文件 `filename` 进行解码操作，其中对于 vial 是否进行特殊处理取决于 `action` 的值。

    - 如果 `action` 是 "csv"，则从 `params` 中获取最后一次生成的 CSV 文件路径，并使用 `settings.upload` 方法将其上传到指定位置。

    4. 最后输出页面尾部模板。
    """

    # 获取 platebarcode
    platebarcode = params.get("platebarcode")

    # 输出页面头部，根据模式不同显示不同的内容
    print(
        head_template
        % {
            "title": "RackScanner 3.0",
            "verb": {"scanner": "Scan", "demo": "Demo", "read_last": "Read"}[
                settings.mode
            ],
            "barcode": (
                '<input name="platebarcode" placeholder="Plate barcode" autofocus></input>'
                if settings.mode == "scanner"
                else ""
            ),
        }
    )

    # 获取 action 类型
    action = params.get("action")

    # 根据 action 类型执行相应操作
    if action in ("rack", "vial"):
        # 在 demo 模式下使用预定义的样本图片
        if settings.mode == "demo":
            filename = (
                "resources/rack_96_sample.bmp"
                if action == "rack"
                else "resources/vial_1ml_sample.bmp"
            )
        # 在 scanner 模式下创建实际扫描文件名并进行扫描
        elif settings.mode == "scanner":
            filename = create_filename(action, platebarcode)
            if action == "vial":
                scanner_controller.scan(filename, right=0.5, bottom=0.5)
            else:
                scanner_controller.scan(filename)
        # 其他情况（如 read_last）读取上次扫描的图片
        else:
            filename = last_image(settings.images_dir)

        # 解码扫描后的图像文件
        decode(filename, action == "vial")
    # 如果 action 是 "csv"，则上传最后一次生成的 CSV 文件
    elif action == "csv":
        settings.upload(params["last_csv"])

    # 输出页面尾部
    print(foot_template)


def create_filename(rack_or_vial: str, barcode: Optional[str] = None) -> str:
    """
    create_filename 函数用于根据输入的 `rack_or_vial` 标识（如架子编号或试管编号）和可选的条形码信息生成一个文件名。这个文件名包含了当前时间戳以及指定的标识，并将其保存为 .bmp 格式的图片文件。

    参数:
        rack_or_vial (str): 代表实验室设备（例如：架子、试管架等）或者单个试管的唯一标识字符串。
        barcode (str, optional): 可选的条形码字符串，若提供，则会包含在生成的文件名中以增加额外的信息标识。默认值为 None。

    返回值:
        str: 根据给定参数格式化后生成的文件路径名，其格式如下：
            bmp/rack_or_vial-YYYY-MM-DD-HH-MM-SS_bmp.bmp

        若提供了 `barcode`，则格式变为：
            bmp/rack_or_vial-YYYY-MM-DD-HH-MM-SS_barcode_bmp.bmp

    步骤说明：
    1. 获取当前日期和时间并转换为 ISO 格式字符串，然后去除 'T' 符号并用短横线替换冒号，将时间戳添加到 `rack_or_vial` 后面作为基本文件名部分。
    2. 如果提供了 `barcode` 参数，将其附加到基本文件名后面，前后分别加上下划线作为分隔符。
    3. 将完整的文件名前缀与 ".bmp" 扩展名组合，并加入到 "bmp/" 目录路径下，形成最终的文件路径名。
    4. 返回生成的完整文件路径名。
    """
    # 构建基于当前日期时间戳的基本文件名
    base = (
        rack_or_vial
        + "-"
        + datetime.datetime.now().isoformat()[:19].replace("T", "-").replace(":", "-")
    )
    # 如果提供了条形码信息，将其添加至基本文件名
    if barcode is not None:
        base += "_" + barcode
    # 完成文件名，并确定其在 "bmp" 子目录下的具体路径
    filename = os.path.join("bmp", base + ".bmp")
    # 返回生成的文件路径名
    return filename


def write_table(wells: pd.DataFrame):
    """
    write_table 函数将给定的 wells 数据（通常为包含试管或孔板编码信息的数据框）转换成 HTML 格式的表格并打印输出。

    参数:
        wells (pd.DataFrame): 一个二维数据结构，其中列代表孔板的不同列，行代表不同的孔或井位，并且每个单元格包含对应的编码信息。

    步骤说明：
    1. 将 wells 中的 "code" 列进行 unstack 操作，生成一个新的 DataFrame，使得索引变为原来的行标签，而列标签成为新的行标签，形成类似孔板布局的数据格式。
    2. 初始化一个字符串列表 `html`，用于存储构建的 HTML 表格内容。
    3. 构建表头部分，包括列标签所在的行。
    4. 遍历重塑后的 DataFrame 的每一行（即孔板的每一行）和每一列（即孔板的每一列）：
       a. 对于每一行，首先添加一行标签到 HTML 内容中。
       b. 对于该行的每一个编码值，检查是否符合 "\d{10}"（10位数字）的正则表达式。如果符合，则不添加 CSS 类；否则，根据编码值创建一个 CSS 类名，并添加到 `<td>` 标签中。
       c. 将带有或不带有类名的 `<td>` 标签以及编码值添加到 HTML 内容中。
    5. 在所有行遍历完成后，关闭表格标记，并将整个 HTML 表格内容连接成一个字符串。
    6. 最后，通过 print 函数逐行输出构建好的 HTML 表格文本。

    输出：
      - 一个以字符串形式表示的 HTML 表格，展示 wells 数据中的编码信息，同时针对不符合特定格式的编码应用了自定义 CSS 类样式。
    """
    # 将 wells 数据重塑为类似于孔板布局的 DataFrame
    plate = wells["code"].unstack()
    # 初始化 HTML 表格内容列表
    html = ['<table class="plate">']
    # 构造表头行，包括列标题
    html += ["<tr><th>"] + ["<th>%s" % i for i in plate.columns]
    # 遍历孔板的每一行
    for row in plate.index:
        # 添加行标题
        html.append("<tr><th>%s" % row)
        # 遍历该行下的每一列
        for col in plate.columns:
            # 获取当前井位的编码
            code = plate.loc[row, col]
            # 根据编码是否符合指定格式决定是否添加CSS类
            cls = "" if re.match("\d{10}", code) else ' class="%s"' % code
            # 创建并添加<td>元素至HTML内容
            html.append("<td%s>%s" % (cls, code))
    # 结束表格
    html.append("</table>")
    # 打印拼接后的完整HTML表格字符串
    print("\n".join(html))


def decode(filename: str, vial: bool = False):
    """
    decode 函数用于解码指定图像文件（filename）中的试管或管架条形码信息。
    如果参数 vial 为 True，则针对单个试管进行解码；否则默认处理整个管架。

    参数:
        filename (str): 需要解码的图像文件名。
        vial (bool, optional): 标记是否解码单个试管。默认值为 False，表示处理整个管架。

    步骤说明：
    1. 调用 dm_reader 模块的 read 函数，传入图像文件名和 vial 参数，获取解码后的井位数据（wells）以及数字矩阵图片（dg_pic）。
    2. 使用 write_table 函数将解码后的井位数据转换并输出为 HTML 表格格式。
    3. 使用 matplotlib 的 image 子模块保存数字矩阵图片为 "dg_pic.png"。
    4. 输出 HTML 标签显示解码得到的数字矩阵图片。
    5. 根据原图像文件名生成对应的 CSV 文件名，并将非空方法（method 不等于 empty）的井位编码信息写入该 CSV 文件。
    6. 输出隐藏的表单字段，包含最后生成的 CSV 文件名，以便在后续操作中使用。
    7. 如果 settings 中设置了用户信息，则输出一个提交按钮，允许用户上传已生成的 CSV 文件。
    """
    # 解码图像文件并获取井位数据及数字矩阵图片
    wells, dg_pic = dm_reader.read(filename, vial=vial)
    # 将解码结果输出为HTML表格
    write_table(wells)
    # 保存数字矩阵图片到本地文件 "dg_pic.png"
    mpl.image.imsave("dg_pic.png", dg_pic)
    # 输出HTML标签展示数字矩阵图片
    print('<img id="dg_pic" src="dg_pic.png" />')
    # 构造CSV文件名并保存井位编码信息到CSV文件
    csvfilename = "csv/" + os.path.splitext(filename)[1].replace("bmp", "csv")
    wells.loc[wells["method"] != "empty"].code.to_csv(csvfilename, sep=";")
    # 输出隐藏的表单字段，包含最新生成的CSV文件名
    print('<input id=last_csv name=last_csv value="%s"/>' % (csvfilename))
    # 如果存在用户设置，则输出上传CSV文件的按钮
    if settings.user:
        print('<button type="submit" name="action" value="csv">Upload CSV</button>')


def last_image(dirname: str) -> str:
    """
    last_image 函数用于查找指定目录（dirname）中最新修改时间的图像文件，
    并返回该文件的完整路径。

    参数:
        dirname (str): 需要遍历以查找最新图像文件的目录路径。

    返回值:
        str: 最新修改过的图像文件的完整路径。图像格式可以是 bmp、png、tiff 或 jpeg。
    """
    # 初始化最大修改时间戳和对应的文件名变量
    max_mtime = 0
    max_file = None
    # 遍历给定目录下的所有文件
    for filename in os.listdir(dirname):
        # 构造当前文件的完整路径
        full_path = os.path.join(dirname, filename)
        # 获取当前文件的最后修改时间戳
        mtime = os.path.getmtime(full_path)
        # 检查文件是否为支持的图像格式
        if mtime > max_mtime and filename.split(".")[-1] in [
            "bmp",
            "png",
            "tiff",
            "jpeg",
        ]:
            # 更新最大修改时间戳和对应文件名
            max_mtime = mtime
            max_file = full_path

    # 返回最新修改过的图像文件的完整路径
    return max_file
