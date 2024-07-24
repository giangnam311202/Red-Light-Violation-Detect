import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


# Paths to video and YOLO model
video_path = r"C:\Disk D\Semester3\HK2\Project\detection\Video\Video1.mp4"
# video_path = r"Video_split/Video/Video4.mp4"


model_path = r"C:\Disk D\Semester3\HK2\Project\detection\weights\best.pt"
output_video_path = "output_video1.mp4"  # Đường dẫn đến file video output

# List of class names
classNames = ['bike', 'bus', 'car', 'green_light', 'motobike', 'red_light', 'stop_line', 'truck', 'yellow_light']

# Load YOLO model
model = YOLO(model_path)

# Mở video
cap = cv2.VideoCapture(video_path)

# Kích thước khung hình mới
frame_width = 1280
frame_height = 720
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Thiết lập kích thước khung hình cho video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Khởi tạo VideoWriter để ghi video
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Tracking
tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.3)

limits = [400, 297, 673, 297]
max_conf = 0
max_distance = 0
crossedIDs = set()
notErrorIDs = set()

while True:
    success, img = cap.read()

    # Check if frame is read successfully
    if not success:
        print("Error: Failed to read frame.")
        break

    # Thay đổi kích thước khung hình
    img = cv2.resize(img, (frame_width, frame_height))

    results = model(img, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == "stop_line" and conf > 0.3:
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if (distance > max_distance):
                    max_distance = distance
                    limits = [x1, int((y1 + y2)/2), x2, int((y1 + y2)/2)]
                print(limits)

            if currentClass != "green_light" and currentClass != "red_light" and currentClass != "stop_line" \
                    and currentClass != "yellow_light" and conf > 0.25:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1,
                                   offset=5, colorR=(100, 100, 100))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    results = model(img, stream=True)
    for t in results:
        boxes = t.boxes
        check_red_light = 0

        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "red_light" or currentClass == "yellow_light" or currentClass == "green_light" and conf > 0.3:
                cvzone.cornerRect(img, (x1, y1, w, h), l=5, rt=1, colorR=(255, 0, 255))
                cvzone.putTextRect(img, f'{currentClass}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1, offset=3)

            if currentClass == "red_light" and conf > 0.3 and check_red_light != 2:
                check_red_light = 1

            if currentClass == "yellow_light" or currentClass == "green_light" and conf > 0.3:
                check_red_light = 2

        if check_red_light == 1:
            for result in resultsTracker:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(result)
                w, h = x2 - x1, y2 - y1
                # cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1, offset=10)

                if id not in notErrorIDs:
                    if y2 > limits[1]:
                        notErrorIDs.add(id)
                        # cv2.circle(img, (x1, y1), 5, (0, 255, 0), cv2.FILLED)
                        # cv2.circle(img, (x2, y2), 5, (0, 255, 0), cv2.FILLED)
                if id not in crossedIDs:
                    if y2 < limits[1] and (id in notErrorIDs):
                        crossedIDs.add(id)
                        notErrorIDs.remove(id)
                        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
                        # cv2.circle(img, (x1, y1), 5, (0, 255, 0), cv2.FILLED)
                        # cv2.circle(img, (x2, y2), 5, (0, 255, 0), cv2.FILLED)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        #cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1, offset=1)
        if id in notErrorIDs:
            # cv2.circle(img, (x1, y1), 5, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (cx, y2), 5, (255, 255, 0), cv2.FILLED)

        if id in crossedIDs:
            cvzone.putTextRect(img, f'Violated', (max(35, x1), max(0, cy)),
                               scale=0.8, thickness=1, offset=3, colorR=(0, 0, 255))

    cvzone.putTextRect(img, f' Violated: {len(crossedIDs)}', (50, 50),
                       scale=5, thickness=3, offset=3, colorR=(0, 0, 255))

    out.write(img)  # Ghi khung hình đã xử lý vào video output
    cv2.imshow("Red light violation detect", img)
    # cv2.imshow("ImageRegion", imgRegion)

    if cv2.waitKey(1) == ord("q"):
        break

print(len(crossedIDs))

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()