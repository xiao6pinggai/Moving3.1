"""项目路径初始化（全局唯一配置）。

自动检测运行环境（autodl服务器 / 本地Windows），将项目根目录加入 sys.path。
其他文件只需在文件开头执行以下 bootstrap 即可：

    import os, sys
    _cur = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(_cur, 'path_setup.py')):
        _cur = os.path.dirname(_cur)
    if _cur not in sys.path:
        sys.path.insert(0, _cur)

之后即可正常使用 `from lib.xxx import yyy` 等绝对导入。
"""
import os
import sys

_CANDIDATE_ROOTS = [
    "/root/autodl-tmp/Moving3.1",       # autodl 服务器
    r"E:\NUDT-Master\Academic\Networks\Moving3.1",  # 本地 Windows
]

ROOT_DIR = None
for _r in _CANDIDATE_ROOTS:
    if os.path.isdir(_r):
        ROOT_DIR = _r
        break

if ROOT_DIR is None:
    # 回退：使用本文件所在目录
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 确保根目录在 sys.path 首位
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
