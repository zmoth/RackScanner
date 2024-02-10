import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
import importlib
import io
import urllib.request, urllib.parse, urllib.error
import traceback
import web_app


class RequestHandler(SimpleHTTPRequestHandler):
    """
    RequestHandler _summary_

    Args:
        SimpleHTTPRequestHandler (_type_): _description_

    Returns:
        _type_: _description_
    """

    def do_GET(self):
        """
        do_GET 重写父类的处理 GET 请求方法

        Returns:
            _type_: _description_
        """
        # 检查请求是否来自本地主机（127.0.0.1），如果不是则发送错误信息并返回
        if self.client_address[0] != "127.0.0.1":
            self.send_error(501, "Only local host allowed")
            return False

        # 分割请求路径与查询参数
        path, _, query = self.path.partition("?")

        # 如果请求路径为根目录（"/"），则执行 CGI 处理逻辑
        if path == "/":
            self.do_CGI(query)
        # 否则调用父类的 do_GET 方法进行常规处理
        else:
            return super().do_GET()

    def do_CGI(self, query):
        """
        do_CGI 定义处理 CGI 请求的方法

        Args:
            query (_type_): _description_
        """
        # 解析查询字符串为字典格式
        dic = self.parse_query(query)
        # 保存原始标准输出对象引用
        sdout = sys.stdout
        # 重新加载 web_app 模块以确保最新代码生效
        importlib.reload(module=web_app)
        # 创建一个内存中的字符串IO对象替代标准输出
        stringio = io.StringIO()
        # 将当前的标准输出临时替换为内存中的StringIO对象
        sys.stdout = stringio
        try:
            # 使用解析后的查询参数字典运行 web_app.run 函数
            web_app.run(**dic)
        except Exception:
            # 若运行过程中出现异常，将异常信息格式化后输出到页面中
            print("<pre>%s</pre>" % traceback.format_exc())
        finally:
            # 设置HTTP响应状态码为200，即成功
            self.send_response(200)
            # 设置内容类型为 text/html
            self.send_header("Content-type", "text/html")
            # 结束头部信息设置
            self.end_headers()
            # 将内存中的StringIO对象内容编码后发送给客户端
            self.wfile.write(stringio.getvalue().encode("utf-8"))

            # 恢复标准输出至原始值
            sys.stdout = sdout
            # 关闭内存中的StringIO对象
            stringio.close()

    def parse_query(self, query):
        """
        parse_query 定义一个用于解析查询字符串的方法

        Args:
            query (_type_): _description_

        Returns:
            _type_: _description_
        """
        # 初始化一个空字典来存储解析后的查询参数
        dic = {}

        # 对查询字符串进行URL解码，并替换 "+" 为 " "
        query = urllib.parse.unquote(query)

        # 遍历分割后的查询参数对
        for par in query.replace("+", " ").split("&"):
            # 分割键值对
            kv = par.split("=", 1)
            # 如果键值对包含等号（即有值）
            if len(kv) == 2:
                name, value = kv
                # 如果该键已在字典中存在
                if name in dic:
                    # 获取旧值
                    oldvalue = dic[name]
                    # 如果旧值是字符串，则转换为列表并添加新值
                    if isinstance(oldvalue, str):
                        dic[name] = [oldvalue, value]
                    # 如果已经是列表，则直接追加新值
                    else:
                        dic[name].append(value)
                # 如果键不存在于字典中，则直接添加键值对
                else:
                    dic[name] = value

        # 返回解析后的查询参数字典
        return dic


if __name__ == "__main__":
    # 获取命令行参数，如果没有指定端口号，则默认使用8000
    if len(sys.argv) > 1 and sys.argv[1:]:
        port = int(sys.argv[1])  # 将第一个命令行参数转换为整数作为服务器监听的端口号
    else:
        port = 8000  # 如果没有提供端口号，则设置为默认值8000
    # 设置服务器地址和端口
    server_address = ("127.0.0.1", port)
    # 创建一个HTTPServer实例，绑定到本地主机与指定端口，并使用自定义的RequestHandler处理请求
    httpd = HTTPServer(server_address, RequestHandler)
    # 获取服务器套接字名称（IP地址和端口号）
    sa = httpd.socket.getsockname()
    # 输出服务器正在监听的地址和端口号
    print("Serving HTTP on", sa[0], "port", sa[1], "...")
    # 开始无限循环，以持续监听并处理客户端的HTTP请求
    httpd.serve_forever()
