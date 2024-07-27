import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt

# 初始化 Mediapipe Pose 模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# 字体文件路径（根据实际路径进行修改）
font_path = "msyh.ttc"


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
            img = put_text_with_custom_font(img, "有人摔倒了，危险！！", (15, 250))

    except Exception as e:
        print(f"Error in fall detection: {str(e)}")


def process_video():
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


if __name__ == "__main__":
    process_video()
