import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]  # 1st frame result

    persons = []
    vehicles = []

    if results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls[0].item())  # Class ID
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates

            if cls == 0:  # Person
                persons.append((x1, y1, x2, y2))
            elif cls in [2, 3, 5, 7]:  # Vehicle classes
                vehicles.append((x1, y1, x2, y2))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    total = len(persons) + len(vehicles)

    # Light logic
    if total >= 1:
        color, text = (0, 0, 255), "RED"
    else:
        color, text = (0, 255, 0), "GREEN"

    # Info on frame
    cv2.putText(frame, f'People: {len(persons)}  Vehicles: {len(vehicles)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f'Light: {text}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Signal light display
    signal = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(signal, (100, 100), 50, color, -1)

    cv2.imshow("Smart Traffic Light", signal)
    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
