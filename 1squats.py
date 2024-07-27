import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# 计算角度函数
def calculate_angle(p1, p2, p3):
    a = np.array([p1.x, p1.y])  # 第一个点
    b = np.array([p2.x, p2.y])  # 第二个点
    c = np.array([p3.x, p3.y])  # 第三个点

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# 处理帧函数
def process_frame(frame):
    img_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_RGB)

    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark

        # 关键点索引
        LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
        RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
        LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
        RIGHT_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE.value
        LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value
        RIGHT_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE.value
        LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
        RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value
        NOSE = mp_pose.PoseLandmark.NOSE.value

        # 手心与脚踝的x轴位置差
        left_wrist_ankle_diff = abs(landmarks[LEFT_WRIST].x - landmarks[LEFT_ANKLE].x)
        right_wrist_ankle_diff = abs(landmarks[RIGHT_WRIST].x - landmarks[RIGHT_ANKLE].x)

        # 背部是否挺直
        back_angle_left = calculate_angle(landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP], landmarks[LEFT_ANKLE])
        back_angle_right = calculate_angle(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP], landmarks[RIGHT_ANKLE])

        # 头是否向上抬起
        head_angle = calculate_angle(landmarks[NOSE], landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])

        # 两脚间距是否略大于肩膀
        shoulder_width = abs(landmarks[LEFT_SHOULDER].x - landmarks[RIGHT_SHOULDER].x)
        feet_width = abs(landmarks[LEFT_ANKLE].x - landmarks[RIGHT_ANKLE].x)

        # 膝关节是否内扣
        left_knee_ankle_diff = abs(landmarks[LEFT_KNEE].x - landmarks[LEFT_ANKLE].x)
        right_knee_ankle_diff = abs(landmarks[RIGHT_KNEE].x - landmarks[RIGHT_ANKLE].x)

        # 大腿是否平行于地面
        left_thigh_angle = calculate_angle(landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
        right_thigh_angle = calculate_angle(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])

        # 在图像上显示文本信息
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font_path = "msyh.ttc"  # 这里指定你的字体路径
        font = ImageFont.truetype(font_path, 20)


        if feet_width > shoulder_width:
            draw.text((25, 110), "两脚间距略大于肩膀", font=font, fill=(0, 0, 0))

        if left_knee_ankle_diff < 0.1 or right_knee_ankle_diff < 0.1:
            draw.text((25, 140), "危险！膝关节内扣", font=font, fill=(0, 0, 0))

        if left_thigh_angle < 80 and left_thigh_angle > 100 and right_thigh_angle < 80 and right_thigh_angle > 100:
            draw.text((25, 170), "大腿平行于地面", font=font, fill=(0, 0, 0))
        if (feet_width < shoulder_width) and (left_knee_ankle_diff > 0.1 or right_knee_ankle_diff < 0.1) and left_thigh_angle > 80 and left_thigh_angle < 100 and right_thigh_angle > 80 and right_thigh_angle < 100:
            draw.text((25, 200), "动作正确！", font=font, fill=(0, 80,100))

        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return frame


def process_video():
    cap = cv2.VideoCapture(0)  # 使用摄像头
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = process_frame(frame)
        frame = cv2.resize(frame, (1000, 750))
        cv2.imshow('my_window', frame)

        if cv2.waitKey(1) in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
            break

    cap.release()
    cv2.destroyAllWindows()


# 调用函数处理视频
process_video()


import cv2  # cv2是python中计算机视觉库OpenCV的一个模块，全称是Open Source Computer Vision Library（开放源代码计算机视觉库）
from PyQt5 import QtCore, QtGui, QtWidgets
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
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

        # 关键点索引
        LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
        RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
        LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
        RIGHT_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE.value
        LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value
        RIGHT_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE.value
        LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
        RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value
        NOSE = mp_pose.PoseLandmark.NOSE.value

        # 手心与脚踝的x轴位置差
        left_wrist_ankle_diff = abs(landmarks[LEFT_WRIST].x - landmarks[LEFT_ANKLE].x)
        right_wrist_ankle_diff = abs(landmarks[RIGHT_WRIST].x - landmarks[RIGHT_ANKLE].x)

        # 背部是否挺直
        back_angle_left = calculate_angle(landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP], landmarks[LEFT_ANKLE])
        back_angle_right = calculate_angle(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP], landmarks[RIGHT_ANKLE])

        # 头是否向上抬起
        head_angle = calculate_angle(landmarks[NOSE], landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])

        # 两脚间距是否略大于肩膀
        shoulder_width = abs(landmarks[LEFT_SHOULDER].x - landmarks[RIGHT_SHOULDER].x)
        feet_width = abs(landmarks[LEFT_ANKLE].x - landmarks[RIGHT_ANKLE].x)

        # 膝关节是否内扣
        left_knee_ankle_diff = abs(landmarks[LEFT_KNEE].x - landmarks[LEFT_ANKLE].x)
        right_knee_ankle_diff = abs(landmarks[RIGHT_KNEE].x - landmarks[RIGHT_ANKLE].x)

        # 大腿是否平行于地面
        left_thigh_angle = calculate_angle(landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
        right_thigh_angle = calculate_angle(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])

        messages = []
        messages2=[]
        messages3=[]
        draw.text((25, 20), "请侧向站立", font=font, fill=(0, 0, 0))

        if left_knee_ankle_diff < 0.1 or right_knee_ankle_diff < 0.1:
            messages.append("危险！膝关节内扣")
            draw.text((25, 50), "危险！膝关节内扣", font=font, fill=(0, 0, 0))

        if left_thigh_angle < 80 and left_thigh_angle > 100 and right_thigh_angle < 80 and right_thigh_angle > 100:
            draw.text((25, 80), "大腿未平行于地面", font=font, fill=(0, 0, 0))
            messages2.append("大腿未平行于地面")

        if (left_knee_ankle_diff > 0.1 and right_knee_ankle_diff > 0.1) and (left_thigh_angle > 80 and left_thigh_angle < 100 and right_thigh_angle > 80 and right_thigh_angle < 100):
            messages3.append("动作正确")
            draw.text((25, 170), "动作正确", font=font, fill=(0, 0, 0))

        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"Error in check_posture function: {str(e)}")

    return img, messages,messages2,messages3