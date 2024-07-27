import cv2
import mediapipe as mp
import numpy as np

# 初始化MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 打开视频流
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 翻转图像，以确保结果非镜像
    frame = cv2.flip(frame, 1)

    # 转换颜色空间
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 处理图像并检测姿势
    results = pose.process(image_rgb)

    # 获取图像尺寸
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2

    if results.pose_landmarks:
        # 计算人体中心点（髋部的中心点）
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        center_body_x = int((left_hip.x + right_hip.x) / 2 * width)
        center_body_y = int((left_hip.y + right_hip.y) / 2 * height)

        # 在图像上绘制中心点
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # 画面中心
        cv2.circle(frame, (center_body_x, center_body_y), 5, (255, 0, 0), -1)  # 人体中心

        # 计算距离
        distance = np.sqrt((center_body_x - center_x) ** 2 + (center_body_y - center_y) ** 2)
        cv2.putText(frame, f'Distance: {int(distance)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Center: ({center_body_x - center_x}, {center_y - center_body_y})', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 绘制连接线
        cv2.line(frame, (center_x, center_y), (center_body_x, center_body_y), (0, 255, 0), 2)

    # 显示图像
    cv2.imshow('MediaPipe Pose', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
