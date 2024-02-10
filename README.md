# <img align="left" height="40px" src="resources/logo.png?raw=true"/> RackScanner

Data Matrix冻存盒底码扫码的应用。测试使用的是Thermo官方的板架盒冻存管。可以读取96孔板架，24孔板架和单管。

![vial_1ml_sample](resources/vial_1ml_sample.bmp)
![rack_96_sample](resources/rack_96_sample.bmp)
![rack_24_sample](resources/rack_24_sample.bmp)


RackScanner能够从可配置的目录读取图像，或者直接操作扫描仪。目前，它仅支持一种类型的扫描仪——**[Avision AVA6平板扫描仪](http://www.avision.net.cn)**。这款扫描仪仅配备Windows 32位TWAIN驱动程序。虽然要使其兼容其他扫描仪并非特别困难，但为了确保正常工作，所使用的扫描仪需要能够对准试管底部所在平面，这个平面距离扫描仪表面大约为2毫米。此外，小型扫描仪格式也是一个优势——AVA6在这方面采用的是A6尺寸。


RackScanner首先通过模式匹配来定位架子上的孔位，并识别出架子的类型。接着，它会确定哪些孔是空的，哪些孔内装有带条形码的试管。接下来，它将使用优秀的高速[OpenCV](http://opencv.org)库中的三种不同算法尝试定位和数字化条形码。解码过程则由[libdmtx](http://libdmtx.sourceforge.net)完成，同时在前面尝试定位条形码失败的情况下，libdmtx也作为备用方案提供服务。

最初的RackScanner在2011年由[jindrichjindrich](https://github.com/jindrichjindrich)开发，并自那时起在[CZ-OPENSCREEN](https://openscreen.cz/en)得到应用。当前的更新源于Thermo Scientific推出的新款试管设计，其采用了圆形Data Matrix模块，libdmtx在读取这类模块时遇到了问题。在这个过程中，我们发现了[Scantelope](https://github.com/dmtaub/scantelope)，它与RackScanner有着相似的目标，并启发了我们的一种算法设计。然而，Scantelope并未解决重新设计后的条形码试管读取问题。

RackScanner已在Linux和Windows系统上进行了测试。

我们非常乐于收到有关RackScanner在不同扫描仪及板/试管类型上的性能反馈。

RackScanner采用[MIT许可协议](https://opensource.org/licenses/MIT)发布。

## 安装指南：
- 从Continuum Analytics安装miniconda（如果需要使用AVA6 TWAIN驱动程序，请选择32位版本）
- 在终端或Anaconda命令提示符（Windows）中执行以下命令：
```
git clone https://github.com/michalkahle/RackScanner.git
cd RackScanner/install
conda env create --file conda_env.yaml
conda activate rackscanner3
```
- 对于Linux系统，安装libdmtx和pylibdmtx：
```
sudo apt install libdmtx0a
pip install pylibdmtx
```
- 对于Windows系统，通过pip安装二进制包：
```
pip install pydmtx-0.7.4b1-cp27-none-win32.whl
pip install twain-1.0.4-cp27-cp27m-win32.whl
vcredist_x86.exe
```
- 切换到父目录并运行HTTP服务器：
```
cd ..
python http_server.py
```
- 打开浏览器访问http://localhost:8000/，在演示模式下测试功能
- 复制`settings_template.py`文件创建一个`settings.py`
- 在`settings.py`文件中，将操作模式更改为'scanner'或'read_last'中的任一种

## 在Windows中安装AVA6驱动程序：
- 重启系统，并关闭驱动程序签名强制检查！
- 使用光盘上的原始安装程序，安装TWAIN和W??驱动程序（可能不是必需的）
- 从设备管理器启动驱动更新过程
- 手动选择驱动程序 -> 点击“浏览我的计算机以查找驱动程序”
- 选择Avision Scanner\Driver\TWAIN目录下的安装文件夹
- 此时应会出现关于安装未签名驱动程序的警告。继续进行安装。
- 断开并重新连接USB线缆。现在扫描仪应该可以正常工作了。
- 注意：该驱动程序仅支持32位TWAIN驱动！

二进制文件来源：
- twain‑1.0.4‑cp27‑cp27m‑win32.whl: http://www.lfd.uci.edu/~gohlke/pythonlibs/#twainmodule
- pydmtx-0.7.4b1-cp27-none-win32.whl: https://github.com/NaturalHistoryMuseum/dmtx-wrappers
