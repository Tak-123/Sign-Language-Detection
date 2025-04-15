import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import mediapipe as mp
import os

# Initialize camera
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

# Folder to save captured images
folder = r"C:/Users/Kabir/Sign-Language-detection/Data/Hello"

# Create folder if it doesn't exist
os.makedirs(folder, exist_ok=True)

while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to read from camera")
        continue

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create white background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Get image dimensions
        height, width, _ = img.shape

        # Safe crop boundaries
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(width, x + w + offset)
        y2 = min(height, y + h + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            print("⚠️ Skipping empty crop.")
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Show cropped and final white image
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    # Show original webcam feed
    cv2.imshow('Image', img)

    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        filename = f'{folder}/Image_{time.time()}.jpg'
        cv2.imwrite(filename, imgWhite)
        print(f"✅ Saved: {filename} ({counter})")

    elif key == ord("q"):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
