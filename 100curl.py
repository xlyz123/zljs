import cv2
import mediapipe as mp
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 初始化 MediaPipe Pose 模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_angle(p1, p2, p3):
    a = np.array([p1.x, p1.y])  # 第一个点的坐标
    b = np.array([p2.x, p2.y])  # 第二个点的坐标
    c = np.array([p3.x, p3.y])  # 第三个点的坐标

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def curl(landmarks, img):
    try:
        # 将OpenCV图像转换为Pillow图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 创建一个可绘制对象
        draw = ImageDraw.Draw(img_pil)

        # 加载自定义字体
        font_path = "msyh.ttc"  # 这里指定你的字体路径
        font = ImageFont.truetype(font_path, 20)

        # 在图像上显示文本
        draw.text((25, 20), "请侧向站立", font=font, fill=(0, 0, 0))

        # 示例关键点索引：假设肩部、肘部和手腕的索引
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

        # 显示角度
        draw.text((25, 50), f'Left Arm Angle: {left_upper_arm_angle:.2f}', font=font, fill=(0, 0, 0))
        draw.text((25, 100), f'Right Arm Angle: {right_upper_arm_angle:.2f}', font=font, fill=(0, 0, 0))

        # 判断姿势是否正确
        if left_upper_arm_angle > 60 or right_upper_arm_angle > 60:
            draw.text((25, 125), '动作错误！大臂请紧贴躯干', font=font, fill=(0, 0, 0))
        if left_elbow_angle > 60 or right_elbow_angle > 60:
            draw.text((25, 150), '请弯曲小臂完成动作', font=font, fill=(0, 0, 0))
        if (left_elbow_angle < 60 or right_elbow_angle < 60)and(left_upper_arm_angle < 40 or right_upper_arm_angle < 40):
            draw.text((25, 175), '动作正确', font=font, fill=(0, 0, 0))

        # 绘制骨骼点
        for idx, landmark in enumerate(landmarks):
            cx, cy = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

        # 将Pillow图像转换回OpenCV图像
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"Error in push-up detection: {str(e)}")

    return img

def process_frame(frame):
    # 将图像转换为 RGB 格式
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 处理图像并检测姿势
    results = pose.process(image)
    # 提取关键点
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        return frame, landmarks
    return frame, None

def process_video():
    cap = cv2.VideoCapture(0)  # 使用摄像头
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, landmarks = process_frame(frame)
        if landmarks:
            frame = curl(landmarks, frame)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 调用函数处理视频
process_video()
