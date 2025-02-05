import cv2
import os
from datetime import datetime


def capture_frames(video_path, output_folder, interval=0.25):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            image_name = f"{timestamp}.jpg"
            image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(image_path, frame)
            print(f"Saved frame at {timestamp}")

        frame_count += 1

    cap.release()
    print("Done")


# Usage example
video_path = r'C:\Users\dhure\Downloads\Vid12 DeepFake.mp4'  # Replace with your video file path
output_folder = r'C:\Users\dhure\Desktop\Virtual Environment\screenshots\vid12 DeepFake'  # Replace with your desired output folder

capture_frames(video_path, output_folder)