from setuptools import setup

setup(
    name="RackScanner",
    version="0.1",
    description="用于读取带有DataMatrix二维码的冻存盒。",
    url="http://github.com/michalkahle/RackScanner",
    author="Michal Kahle",
    author_email="michalkahle@gmail.com",
    license="MIT",
    py_modules=["dm_reader", "http_server", "scanner_controller", "web_app"],
    zip_safe=False,
)
