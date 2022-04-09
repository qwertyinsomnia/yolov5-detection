# -*- coding: utf-8 -*-
"""
运行本项目需要安装的库：
    opencv-contrib-python 4.5.1.48
    PyQt5 5.15.2
    scikit-learn 0.22
    numba 0.53.0
    imutils 0.5.4
    filterpy 1.4.5

点击运行主程序runMain.py，程序所在文件夹路径中请勿出现中文
"""
# -*- coding: utf-8 -*-
# 车辆行人等多目标检测及跟踪系统主程序
# @Time    : 2021/3/13 10:29
# @Author  : sixuwuxian
# @Email   : sixuwuxian@aliyun.com
# @blog    : wuxian.blog.csdn.net
# @Software: PyCharm
import os
import warnings

from DetectionTracking import Ui_MainWindow
from sys import argv, exit
from PyQt5.QtWidgets import QApplication, QMainWindow

if __name__ == '__main__':
    # 忽略警告
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings(action='ignore')
    app = QApplication(argv)

    window = QMainWindow()
    ui = Ui_MainWindow(window)

    window.show()
    exit(app.exec_())
