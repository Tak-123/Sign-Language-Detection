import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("C:/Users/Kabir/Downloads/converted_keras/keras_model.h5", "C:/Users/Kabir/Downloads/converted_keras/labels.txt")
offset = 20
imgSize = 300
counter = 0

labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize



        prediction, index = classifier.getPrediction(imgWhite, draw=False)


        cv2.rectangle(imgOutput, (x - offset, y - offset - 30 * len(labels)),
                      (x + 200, y - offset), (255, 255, 255), cv2.FILLED)

        
        for i, prob in enumerate(prediction):
            label_text = f"{labels[i]}: {prob * 100:.2f}%"
            color = (0, 255, 0) if i == index else (0, 0, 0)
            cv2.putText(imgOutput, label_text,
                        (x - offset, y - offset - 10 - 25 * (len(labels) - i)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)

    # Check if 'q' is pressed to exit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
