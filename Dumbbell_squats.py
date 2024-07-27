import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time

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


def process_frame(img):
    h, w = img.shape[0], img.shape[1]
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_RGB)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        cv2.putText(img, 'No Person', (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)
        print('No person detected in the image.')

    end_time = time.time()
    FPS = 1 / (end_time - start_time)
    cv2.putText(img, 'FPS: ' + str(int(FPS)), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
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

        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        NOSE = 0
        LEFT_EYE = 1
        RIGHT_EYE = 2

        left_wrist_x = landmarks[LEFT_WRIST][1]
        right_wrist_x = landmarks[RIGHT_WRIST][1]
        left_ankle_x = landmarks[LEFT_ANKLE][1]
        right_ankle_x = landmarks[RIGHT_ANKLE][1]

        wrist_ankle_diff_left = abs(left_wrist_x - left_ankle_x)
        wrist_ankle_diff_right = abs(right_wrist_x - right_ankle_x)

        left_back_angle = calculate_angle(landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP], landmarks[LEFT_ANKLE])
        right_back_angle = calculate_angle(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP], landmarks[RIGHT_ANKLE])

        head_angle_left = calculate_angle(landmarks[LEFT_EYE], landmarks[NOSE], landmarks[LEFT_SHOULDER])
        head_angle_right = calculate_angle(landmarks[RIGHT_EYE], landmarks[NOSE], landmarks[RIGHT_SHOULDER])

        shoulder_width = calculate_distance(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
        ankle_width = calculate_distance(landmarks[LEFT_ANKLE], landmarks[RIGHT_ANKLE])

        draw.text((25, 20), "请侧向站立", font=font, fill=(0, 0, 0))

        if wrist_ankle_diff_left > 100 or wrist_ankle_diff_right > 100:
            draw.text((25, 50), "杠铃保存在脚心正上方", font=font, fill=(0, 0, 0))

        if ankle_width > shoulder_width * 1.5:
            draw.text((25, 80), "两脚间距略大于肩膀", font=font, fill=(0, 0, 0))
        if ankle_width < shoulder_width:
            draw.text((25, 80), "两脚间距不足", font=font, fill=(0, 0, 0))

        if (wrist_ankle_diff_left <= 100 and wrist_ankle_diff_right <= 100) and (ankle_width > shoulder_width and ankle_width < shoulder_width*1.5):
            draw.text((25, 170), "动作正确", font=font, fill=(0, 0, 0))

        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"Error in check_posture function: {str(e)}")

    return img


def process_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = process_frame(frame)
        coordinate = get_pose(frame)
        if coordinate:
            frame = check_posture(coordinate, img=frame)

        frame = cv2.resize(frame, (1000, 750))
        cv2.imshow('my_window', frame)

        if cv2.waitKey(1) in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()


start_time = time.time()
process_video()
