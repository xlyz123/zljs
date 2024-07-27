import sys
from PyQt5 import QtCore, QtWidgets, QtCore
from jsnu2 import Ui_Dialog
from main_nu import *
from jkjk2 import Ui_Dialog as Ui_Dialog_jk
from kf_sb import Ui_QMainWindow as Ui_QMainWindow_sb
from kfnu2 import Ui_Dialog as Ui_Dialog_kf
from kf_xz import Ui_QMainWindow as Ui_QMainWindow_xz
from kf_jb import Ui_QMainWindow1 as Ui_QMainWindow_jb
from js_sd import Ui_QMainWindow as Ui_QMainWindow_sd
from jsnu2 import Ui_Dialog as Ui_Dialog_js
from js_tj import Ui_QMainWindow as Ui_QMainWindow_tj
from js_wj import Ui_QMainWindow as Ui_QMainWindow_wj
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton,QWidget
from js_xs import Ui_QMainWindow as Ui_QMainWindow_xs
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog
import sys

class parentWindow_mu1(QMainWindow):
    def __init__(self):
        super().__init__()
        self.main_ui = Ui_MainWindow()
        self.main_ui.setupUi(self)

class childWindow_mu1(QDialog):
    def __init__(self):
        super().__init__()
        self.child = Ui_Dialog()
        self.child.setupUi(self)

class childWindow1_mu1(QDialog):
    def __init__(self):
        super().__init__()
        self.child = Ui_Dialog_kf()
        self.child.setupUi(self)

class childWindow2_mu1(QDialog):
    def __init__(self):
        super().__init__()
        self.child = Ui_Dialog_jk()
        self.child.setupUi(self)

class parentWindow_nu2(QMainWindow):
    def __init__(self):
        super().__init__()
        self.main_ui = Ui_Dialog_kf()
        self.main_ui.setupUi(self)

# class childWindow_nu2(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.child = Ui_QMainWindow_sb()
#         self.child.setupUi(self)
#
# class childWindow1_nu2(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.child = Ui_QMainWindow_xz()
#         self.child.setupUi(self)
# #
# class childWindow2_nu2(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.child = Ui_QMainWindow_jb()
#         self.child.setupUi(self)

class parentWindow_nu3(QMainWindow):
    def __init__(self):
        super().__init__()
        self.main_ui = Ui_Dialog_js()
        self.main_ui.setupUi(self)

# class childWindow_nu3(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.child = Ui_QMainWindow_sd()
#         self.child.setupUi(self)

#
#class childWindow2_nu3(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.child = Ui_QMainWindow_wj()
#         self.child.setupUi(self)
#
class childWindow3_nu3(QMainWindow):
    def __init__(self):
        super().__init__()
        self.child = Ui_QMainWindow_xs()
        self.child.setupUi(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = parentWindow_mu1()
    child_mu1 = childWindow_mu1()
    child1_mu1 = childWindow1_mu1()
    child2_mu1 = childWindow2_mu1()

    window_nu2 = parentWindow_nu2()
    child_nu2 = Ui_QMainWindow_sb()
    child1_nu2 = Ui_QMainWindow_xz()
    child2_nu2 = Ui_QMainWindow_jb()

    window_nu3 = parentWindow_nu3()
    child_nu3 = Ui_QMainWindow_sd()
    #child1_nu3 = childWindow1_nu3()
    child2_nu3 = Ui_QMainWindow_wj()
    child3_nu3 = childWindow3_nu3()

    btn = window.main_ui.toolButton
    btn.clicked.connect(window_nu3.show)
    btn1 = window.main_ui.toolButton_2
    btn1.clicked.connect(window_nu2.show)
    btn2 = window.main_ui.toolButton_3
    btn2.clicked.connect(child2_mu1.show)

    btn_nu2 = window_nu2.main_ui.toolButton_2
    btn_nu2.clicked.connect(child_nu2.show)
    btn1_nu2 = window_nu2.main_ui.toolButton
    btn1_nu2.clicked.connect(child1_nu2.show)
    btn2_nu2 = window_nu2.main_ui.toolButton_3
    btn2_nu2.clicked.connect(child2_nu2.show)

    btn_nu3 = window_nu3.main_ui.pushButton
    btn_nu3.clicked.connect(child_nu3.show)
    btn1_nu3 = window_nu3.main_ui.pushButton_2
    #btn1_nu3.clicked.connect(child1_nu3.show)
    btn2_nu3 = window_nu3.main_ui.pushButton_3
    btn2_nu3.clicked.connect(child2_nu3.show)
    btn3_nu3 = window_nu3.main_ui.pushButton_4
    btn3_nu3.clicked.connect(child3_nu3.show)

    window.show()
    sys.exit(app.exec_())
