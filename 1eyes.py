import cv2
import mediapipe as mp

# Initialize MediaPipe face detection model
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Open a local video file
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error reading frame")
        break

    # Convert BGR image to RGB format because MediaPipe face detection model requires RGB input
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run face detection model
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, bbox, (0, 255, 0), 2)

            # Print bounding box coordinates
            print(f"Bounding Box Coordinates: {bbox}")

    # Calculate and print frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frame rate: {fps}")
    # Check for 'q' key press to exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
