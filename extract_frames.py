import cv2
import os

def extract_frames(video_path, output_folder, max_frames=None):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: cannot open video.")
        return

    count = 0

    while True:
        if max_frames is not None and count >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        cv2.imwrite(f"{output_folder}/frame_{count:03d}.png", frame)
        count += 1

    cap.release()
    print(f"{count} frames extraites dans {output_folder}/")

extract_frames("Video1.mp4", "frames/", max_frames=None)
