import speech_recognition as sr
import os
import time
from playsound import playsound
##################
import numpy as np
import math
import cv2
import mediapipe as mp
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt

# 设置MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

###############################################################################################################
#以下为弯举评测
def put_text_with_custom_font(img, text, pos, font_size=0.5, color=(255, 0, 255), thickness=2):
    # 定义自定义字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2

    # 使用自定义字体显示中文
    cv2.putText(img, text, pos, font, font_scale, color, thickness, lineType=cv2.LINE_AA)


def look_img(img):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()


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
    return []


def process_frame(img):
    global start_time
    start_time = time.time()
    h, w = img.shape[0], img.shape[1]
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_RGB)

    coordinate = []  # 定义一个空的 coordinate 变量

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for i in range(33):
            cx = int(results.pose_landmarks.landmark[i].x * w)
            cy = int(results.pose_landmarks.landmark[i].y * h)
            single = [i, cx, cy]
            coordinate.append(single)

            radius = 5
            if i == 0:
                img = cv2.circle(img, (cx, cy), 1, (0, 0, 255), -1)
            # 省略其他绘制圆形的部分

        xmin, ymin = min([x[1] for x in coordinate]), min([x[2] for x in coordinate])
        xmax, ymax = max([x[1] for x in coordinate]), max([x[2] for x in coordinate])
        boxW = xmax - xmin
        boxH = ymax - ymin
        cv2.rectangle(img, (xmin - 30, ymin - 30), (xmin + boxW + 20, ymin + boxH + 20), (255, 0, 255), 2)
    else:
        img = cv2.putText(img, 'No Person', (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)

    end_time = time.time()
    FPS = 1 / (end_time - start_time)
    img = cv2.putText(img, 'FPS  ' + str(int(FPS)), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return img, coordinate


def push_up(coordinate, img):
    try:
        # 在图像上显示文本
        img = cv2.putText(img, "请侧向站立", (25, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # 获取关键点坐标
        ljb_x, ly_x, lxg_x = coordinate[11][1], coordinate[23][1], coordinate[25][1]
        ljb_y, ly_y, lxg_y = coordinate[11][2], coordinate[23][2], coordinate[25][2]
        rjb_x, ry_x, rxg_x = coordinate[12][1], coordinate[24][1], coordinate[26][1]
        rjb_y, ry_y, rxg_y = coordinate[12][2], coordinate[24][2], coordinate[26][2]
        lxb_x, lw_x, lzz_x = coordinate[14][1], coordinate[16][1], coordinate[20][1]
        lxb_y, lw_y, lzz_y = coordinate[14][2], coordinate[16][2], coordinate[20][2]
        rxb_x, rw_x, rzz_x = coordinate[13][1], coordinate[15][1], coordinate[19][1]
        rxb_y, rw_y, rzz_y = coordinate[13][2], coordinate[15][2], coordinate[19][2]

        # 计算角度
        ls_xv, ls_yv = ljb_x - ly_x, ljb_y - ly_y
        lx_xv, lx_yv = lxg_x - ly_x, lxg_y - ly_y
        rs_xv, rs_yv = rjb_x - ry_x, rjb_y - ry_y
        rx_xv, rx_yv = rxg_x - ry_x, rxg_y - ry_y
        cos_l1 = (ls_xv * lx_xv + ls_yv * lx_yv) / (
                (math.sqrt(ls_xv ** 2 + ls_yv ** 2)) * (math.sqrt(lx_xv ** 2 + lx_yv ** 2)))
        lp1 = math.degrees(math.acos(cos_l1))
        cos_r1 = (rs_xv * rx_xv + rs_yv * rx_yv) / (
                (math.sqrt(rs_xv ** 2 + rs_yv ** 2)) * (math.sqrt(rx_xv ** 2 + rx_yv ** 2)))
        rp1 = math.degrees(math.acos(cos_r1))

        lb_xv, lb_yv = lxb_x - lw_x, lxb_y - lw_y
        lz_xv, lz_yv = lzz_x - lw_x, lzz_y - lw_y
        rb_xv, rb_yv = rxb_x - rw_x, rxb_y - rw_y
        rz_xv, rz_yv = rzz_x - rw_x, rzz_y - rw_y
        cos_l2 = (lb_xv * lz_xv + lb_yv * lz_yv) / (
                (math.sqrt(lb_xv ** 2 + lb_yv ** 2)) * (math.sqrt(lz_xv ** 2 + lz_yv ** 2)))
        lp2 = math.degrees(math.acos(cos_l2))
        cos_r2 = (rb_xv * rz_xv + rb_yv * rz_yv) / (
                (math.sqrt(rb_xv ** 2 + rb_yv ** 2)) * (math.sqrt(rz_xv ** 2 + rz_yv ** 2)))
        rp2 = math.degrees(math.acos(cos_r2))

        ldb_xv, ldb_yv = ljb_x - lxb_x, ljb_y - lxb_y
        rdb_xv, rdb_yv = rjb_x - rxb_x, rjb_y - rxb_y
        ldx_xv, ldx_yv = lxb_x - lxg_x, lxb_y - lxg_y
        rdx_xv, rdx_yv = rxb_x - rxg_x, rxb_y - rxg_y
        cos_l3 = (ldb_xv * ldx_xv + ldb_yv * ldx_yv) / (
                (math.sqrt(ldb_xv ** 2 + ldb_yv ** 2)) * (math.sqrt(ldx_xv ** 2 + ldx_yv ** 2)))
        lp3 = math.degrees(math.acos(cos_l3))
        cos_r3 = (rdb_xv * rdx_xv + rdb_yv * rdx_yv) / (
                (math.sqrt(rdb_xv ** 2 + rdb_yv ** 2)) * (math.sqrt(rdx_xv ** 2 + rdx_yv ** 2)))
        rp3 = math.degrees(math.acos(cos_r3))

        # 计算平均角度
        lp, rp = round((lp1 + lp2 + lp3) / 3, 2), round((rp1 + rp2 + rp3) / 3, 2)

        # 在图像上显示角度信息
        img = cv2.putText(img, "左臂角度 : " + str(lp), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        img = cv2.putText(img, "右臂角度 : " + str(rp), (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # 判断姿势是否正确
        if (lp > 60) or (rp > 60):
            img = cv2.putText(img, '大臂紧贴躯干', (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        else:
            img = cv2.putText(img, '动作正确，继续加油', (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    except Exception as e:
        print(f"Error in push-up detection: {str(e)}")

    return img

def open_notepad():
    cap = cv2.VideoCapture(0)  # 使用摄像头
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, coordinate = process_frame(frame)
        frame = push_up(coordinate, frame)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


#######################################################################################################
def open_calculator():
    os.system("calc")  # 适用于Windows
    # os.system("gnome-calculator")  # 适用于Linux
#######################################################################################################

def put_text_with_custom_font(img, text, pos, font_size=0.5, color=(255, 0, 255), thickness=2):
    # 定义自定义字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2

    # 使用自定义字体显示中文
    cv2.putText(img, text, pos, font, font_scale, color, thickness, lineType=cv2.LINE_AA)


def process_frame(img):
    h, w = img.shape[0], img.shape[1]
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_RGB)

    coordinate = []

    if results.pose_landmarks:
        for i in range(33):
            cx = int(results.pose_landmarks.landmark[i].x * w)
            cy = int(results.pose_landmarks.landmark[i].y * h)
            single = [i, cx, cy]
            coordinate.append(single)

    return img, coordinate


def put_text_with_custom_font(img, text, pos, font_size=0.5, color=(255, 0, 255), thickness=2):
    # 定义自定义字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2

    # 使用自定义字体显示中文
    cv2.putText(img, text, pos, font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    return img


def fall_detection(coordinate, img):
    try:
        face_point_x, face_point_y = coordinate[0][1], coordinate[0][2]
        lknee_x, lknee_y = coordinate[26][1], coordinate[26][2]

        if face_point_y > lknee_y:
            # 在图像上显示摔倒提示，使用自定义字体
            img = put_text_with_custom_font(img, "老人摔倒了，危险！！", (15, 250))

    except Exception as e:
        print(f"Error in fall detection: {str(e)}")


def open_browser():
    cap = cv2.VideoCapture(0)  # 使用默认摄像头
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, coordinate = process_frame(frame)
        if coordinate:
            fall_detection(coordinate, frame)

        # 显示图像
        cv2.imshow('Frame', frame)

        # 设置退出条件
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




#########################3


# 创建语音识别器实例
recognizer = sr.Recognizer()


# 录制音频并检测唤醒词
def listen_for_wake_word(recognizer, microphone, wake_word="你好"):
    while True:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            print("监听唤醒词...")
            audio = recognizer.listen(source, timeout=3)
            try:
                transcription = recognizer.recognize_google(audio, language="zh-CN")
                print(f"检测到的语音: {transcription}")
                if wake_word in transcription:
                    print("唤醒词检测成功")
                    #playsound("path/to/sound/file.mp3")  # 播放提示音
                    return
            except sr.UnknownValueError:
                print("无法识别语音")
            except sr.RequestError:
                print("API不可用")


# 录制音频并识别命令
def recognize_speech_from_mic(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("请开始说话...")
        audio = recognizer.listen(source, timeout=3)

    response = {"success": True, "error": None, "transcription": None}

    try:
        response["transcription"] = recognizer.recognize_google(audio, language="zh-CN")
    except sr.RequestError:
        response["success"] = False
        response["error"] = "API不可用"
    except sr.UnknownValueError:
        response["error"] = "无法识别语音"

    return response


# 主程序
if __name__ == "__main__":
    microphone = sr.Microphone()

    while True:
        listen_for_wake_word(recognizer, microphone)

        while True:
            speech_recognition_result = recognize_speech_from_mic(recognizer, microphone)
            if speech_recognition_result["success"]:
                print(f"识别结果: {speech_recognition_result['transcription']}")
                command = speech_recognition_result["transcription"]
                if "健身锻炼" in command:
                    open_notepad()
                elif "康复训练" in command:
                    open_calculator()
                elif "健康监控" in command:
                    open_browser()
                elif "退出" in command:
                    print("系统退出...")
                    exit()
                else:
                    print("无法识别指令，请重试。")
            else:
                print(f"错误: {speech_recognition_result['error']}")
