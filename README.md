# object-detection-traffic-light
# Smart Traffic Light using YOLOv8 🚦
This project implements a real-time AI-based traffic signal controller using YOLOv8 (Ultralytics) and OpenCV. It detects people and vehicles from a webcam feed and automatically switches the traffic light between RED and GREEN based on the number of detected objects.


A real-time AI-powered traffic light system that switches between RED and GREEN based on detected people and vehicles.

## Features
- Uses YOLOv8 for object detection
- Detects persons and vehicles
- Simulates signal light based on traffic density
- Real-time camera feed (OpenCV)

## Run It
```bash
pip install -r requirements.txt
python traffic_light.py
