# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'js_sd.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
from PyQt5 import QtWidgets
import cv2
from PyQt5 import QtCore, QtGui
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def get_pose(img):
    try:
        h, w = img.shape[0], img.shape[1]
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_RGB)
        if results.pose_landmarks:
            coordinates = []
            for i in range(33):
                cx = int(results.pose_landmarks.landmark[i].x * w)
                cy = int(results.pose_landmarks.landmark[i].y * h)
                coordinates.append([i, cx, cy])
            return coordinates
        return None
    except Exception as e:
        print(f"Error in get_pose function: {str(e)}")
        return None

def process_frame(img):
    try:
        h, w = img.shape[0], img.shape[1]
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_RGB)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            cv2.putText(img, 'No Person', (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)
            print('No person detected in the image.')
        return img
    except Exception as e:
        print(f"Error in process_frame function: {str(e)}")
        return img

def calculate_angle(a, b, c):
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360.0 - angle
        return angle
    except Exception as e:
        print(f"Error in calculate_angle function: {str(e)}")
        return None

def check_posture(landmarks, img):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font_path = "msyh.ttc"
        font = ImageFont.truetype(font_path, 20)

        # 定义关键点索引
        NOSE = 0
        LEFT_EYE = 1
        RIGHT_EYE = 2
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28

        # 计算头部中点
        head_center = [
            (landmarks[NOSE][1] + landmarks[LEFT_EYE][1] + landmarks[RIGHT_EYE][1]) / 3,
            (landmarks[NOSE][2] + landmarks[LEFT_EYE][2] + landmarks[RIGHT_EYE][2]) / 3
        ]

        # 计算肩部中点
        shoulder_center = [
            (landmarks[LEFT_SHOULDER][1] + landmarks[RIGHT_SHOULDER][1]) / 2,
            (landmarks[LEFT_SHOULDER][2] + landmarks[RIGHT_SHOULDER][2]) / 2
        ]

        # 计算髋部中点
        hip_center = [
            (landmarks[LEFT_HIP][1] + landmarks[RIGHT_HIP][1]) / 2,
            (landmarks[LEFT_HIP][2] + landmarks[RIGHT_HIP][2]) / 2
        ]

        # 计算头部中点、肩部中点和髋部中点构成的角度
        head_shoulder_hip_angle = calculate_angle(head_center, shoulder_center, hip_center)

        # 计算左侧大腿与小腿的角度
        left_leg_angle = calculate_angle(
            [landmarks[LEFT_HIP][1], landmarks[LEFT_HIP][2]],
            [landmarks[LEFT_KNEE][1], landmarks[LEFT_KNEE][2]],
            [landmarks[LEFT_ANKLE][1], landmarks[LEFT_ANKLE][2]]
        )

        # 计算右侧大腿与小腿的角度
        right_leg_angle = calculate_angle(
            [landmarks[RIGHT_HIP][1], landmarks[RIGHT_HIP][2]],
            [landmarks[RIGHT_KNEE][1], landmarks[RIGHT_KNEE][2]],
            [landmarks[RIGHT_ANKLE][1], landmarks[RIGHT_ANKLE][2]]
        )


        messages = []
        messages2=[]
        messages3=[]
        draw.text((25, 20), "请侧向站立", font=font, fill=(0, 0, 0))

        if abs(head_shoulder_hip_angle - 180) > 5:
            messages.append("动作错误！头、肩、髋不在一条直线上")
            draw.text((25, 50), "动作错误！头、肩、髋不在一条直线上", font=font, fill=(0, 0, 0))

        if left_leg_angle < 90 or right_leg_angle < 90:
            messages2.append("动作错误！大腿与小腿夹角小于90度")
            draw.text((25, 80), "动作错误！大腿与小腿夹角小于90度", font=font, fill=(0, 0, 0))

        if (abs(head_shoulder_hip_angle - 180) < 5) and ((left_leg_angle > 85 and right_leg_angle > 100) or (right_leg_angle > 90 and left_leg_angle > 100)):
            messages3.append("动作正确")
            draw.text((25, 170), "动作正确", font=font, fill=(0, 0, 0))

        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"Error in check_posture function: {str(e)}")

    return img, messages,messages2,messages3

class Ui_QMainWindow1(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_QMainWindow1, self).__init__()
        self.setupUi(self)
        self.camera_timer = QTimer()
        self.cap = cv2.VideoCapture(0)
        self.init()

    def setupUi(self, QMainWindow):
        QMainWindow.setObjectName("QMainWindow")
        QMainWindow.resize(803, 595)
        self.centralwidget = QtWidgets.QWidget(QMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.listView = QtWidgets.QListView(self.centralwidget)
        self.listView.setGeometry(QtCore.QRect(0, 0, 811, 551))
        self.listView.setStyleSheet("background-image:url(:/pic29.png)")
        self.listView.setObjectName("listView")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 60, 411, 261))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 340, 411, 51))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(0, 400, 421, 61))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(0, 470, 421, 51))
        self.label_4.setObjectName("label_4")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(520, 420, 100, 40))
        font = QtGui.QFont()
        font.setFamily("阿里巴巴普惠体")
        font.setPointSize(11)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(520, 470, 100, 40))
        font = QtGui.QFont()
        font.setFamily("阿里巴巴普惠体")
        font.setPointSize(11)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        QMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(QMainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 803, 22))
        self.menubar.setObjectName("menubar")
        QMainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(QMainWindow)
        self.statusbar.setObjectName("statusbar")
        QMainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(QMainWindow)
        QtCore.QMetaObject.connectSlotsByName(QMainWindow)

    def retranslateUi(self, QMainWindow):
        _translate = QtCore.QCoreApplication.translate
        QMainWindow.setWindowTitle(_translate("QMainWindow", "MainWindow"))
        self.label.setText(_translate("QMainWindow", "TextLabel"))
        self.label_2.setText(_translate("QMainWindow", "TextLabel"))
        self.label_3.setText(_translate("QMainWindow", "TextLabel"))
        self.label_4.setText(_translate("QMainWindow", "TextLabel"))
        self.pushButton.setText(_translate("QMainWindow", "开始锻炼"))
        self.pushButton_2.setText(_translate("QMainWindow", "暂停锻炼"))

    def init(self):
        self.pushButton.clicked.connect(self.open_camera)
        self.pushButton_2.clicked.connect(self.close_camera)
        self.camera_timer.timeout.connect(self.show_image)

    def open_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.camera_timer.start(40)

    def show_image(self):
        flag, self.image = self.cap.read()
        if not flag:
            print("Failed to capture image from camera.")
            return

        self.image = process_frame(self.image)
        coordinate = get_pose(self.image)
        messages = []
        messages2 = []
        messages3 = []
        if coordinate:
            self.image, messages, messages2, messages3 = check_posture(coordinate, img=self.image)

        image_show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        height, width, channel = image_show.shape
        step = channel * width
        qImg = QImage(image_show.data, width, height, step, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qImg)
        scaled_pixmap = pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio)

        self.label.setPixmap(scaled_pixmap)

        self.label_2.setText("\n".join(messages))
        self.label_3.setText("\n".join(messages2))
        self.label_4.setText("\n".join(messages3))

    def close_camera(self):
        self.camera_timer.stop()
        self.cap.release()
        self.label.clear()
        self.label.setText("TextLabel")

import text10_rc

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_win = Ui_QMainWindow1()
    main_win.show()
    sys.exit(app.exec_())
