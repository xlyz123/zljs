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

def get_pose(img):
    h, w = img.shape[0], img.shape[1]
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_RGB)
    if results.pose_landmarks:
        coordinate = []
        for i in range(33):
            cx = int(results.pose_landmarks.landmark[i].x * w)
            cy = int(results.pose_landmarks.landmark[i].y * h)
            single = [i, cx, cy]
            coordinate.append(single)
        return coordinate
    return None

def process_frame(img):
    h, w = img.shape[0], img.shape[1]
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_RGB)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        cv2.putText(img, 'No Person', (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)
        print('No person detected in the image.')
    return img

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def calculate_angle(p1, p2, p3):
    a = np.array([p1[0], p1[1]])
    b = np.array([p2[0], p2[1]])
    c = np.array([p3[0], p3[1]])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def check_posture(landmarks, img):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font_path = "msyh.ttc"
        font = ImageFont.truetype(font_path, 20)

        # 获取左侧关键点
        LEFT_SHOULDER = 11
        LEFT_ELBOW = 13
        LEFT_WRIST = 15
        RIGHT_SHOULDER = 12
        RIGHT_ELBOW = 14
        RIGHT_WRIST = 16

        # 获取左侧大臂角度
        left_upper_arm_angle = calculate_angle(landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST])
        # 获取右侧大臂角度
        right_upper_arm_angle = calculate_angle(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST])

        # 获取左侧大臂与小臂的夹角
        left_elbow_angle = calculate_angle(landmarks[LEFT_ELBOW], landmarks[LEFT_SHOULDER], landmarks[LEFT_WRIST])
        # 获取右侧大臂与小臂的夹角
        right_elbow_angle = calculate_angle(landmarks[RIGHT_ELBOW], landmarks[RIGHT_SHOULDER], landmarks[RIGHT_WRIST])

        messages = []
        messages2 = []
        messages3 = []
        draw.text((25, 20), "请侧向站立", font=font, fill=(0, 0, 0))

        if left_upper_arm_angle > 60 or right_upper_arm_angle > 60:
            messages.append("动作错误！大臂请紧贴躯干")
            draw.text((25, 50), "动作错误！大臂请紧贴躯干", font=font, fill=(0, 0, 0))

        if left_elbow_angle > 60 or right_elbow_angle > 60:
            draw.text((25, 80), "请弯曲小臂完成动作", font=font, fill=(0, 0, 0))
            messages2.append("请弯曲小臂完成动作")

        if (left_elbow_angle < 60 and right_elbow_angle < 60)and(left_upper_arm_angle < 40 and right_upper_arm_angle < 40):
            messages3.append("动作正确")
            draw.text((25, 170), "动作正确", font=font, fill=(0, 0, 0))

        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"Error in check_posture function: {str(e)}")

    return img, messages, messages2, messages3

class Ui_QMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_QMainWindow, self).__init__()
        self.setupUi(self)
        self.camera_timer = QTimer()
        self.cap = cv2.VideoCapture(0)
        self.init()

    def setupUi(self, QMainWindow):
        QMainWindow.setObjectName("QMainWindow")
        QMainWindow.resize(800, 595)
        self.centralwidget = QtWidgets.QWidget(QMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.listView = QtWidgets.QListView(self.centralwidget)
        self.listView.setGeometry(QtCore.QRect(0, 0, 801, 561))
        self.listView.setStyleSheet("background-image:url(:/pic30.png)")
        self.listView.setObjectName("listView")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 60, 421, 261))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(0, 340, 421, 51))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(0, 400, 421, 61))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(0, 470, 421, 51))
        self.label_4.setObjectName("label_4")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(500, 420, 100, 40))
        font = QtGui.QFont()
        font.setFamily("阿里巴巴普惠体")
        font.setPointSize(11)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(500, 470, 100, 40))
        font = QtGui.QFont()
        font.setFamily("阿里巴巴普惠体")
        font.setPointSize(11)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        QMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(QMainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
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
        self.pushButton_2.setText(_translate("QMainWindow", "暂停锻炼"))
        self.pushButton.setText(_translate("QMainWindow", "开始锻炼"))

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


import text11_rc

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_win = Ui_QMainWindow()
    main_win.show()
    sys.exit(app.exec_())
