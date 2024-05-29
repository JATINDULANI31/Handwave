import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the model
model = load_model('sign_detection_model1.h5')

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 400

# Dictionary mapping label index to sign language characters
labels_dict = {
    0: 'I love you', 1: 'No', 2: 'Please', 3: 'Thank you', 4: 'Yes', 5: 'a', 6: 'b', 7: 'c', 8: 'd', 9: 'e', 10: 'f',
    11: 'g', 12: 'h', 13: 'i', 14: 'j', 15: 'k', 16: 'l', 17: 'm', 18: 'n', 19: 'o', 20: 'p', 21: 'q', 22: 'r', 23: 's',
    24: 't', 25: 'u', 26: 'v', 27: 'w', 28: 'x', 29: 'y', 30: 'z'
}

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        y1, y2 = max(0, y - offset), min(y + h + offset, img.shape[0])
        x1, x2 = max(0, x - offset), min(x + w + offset, img.shape[1])
        imgCrop = img[y1:y2, x1:x2]

        imgCropShape = imgCrop.shape
        print(f"imgCrop shape: {imgCropShape}")

        if imgCropShape[0] == 0 or imgCropShape[1] == 0:
            print("Invalid crop dimensions, skipping this frame.")
            continue

        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                print(f"imgResize shape: {imgResizeShape}")
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                print(f"imgResize shape: {imgResizeShape}")
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Preprocess and predict
            roi_resized = cv2.resize(imgWhite, (224, 224))
            roi_expanded = np.expand_dims(roi_resized, axis=0)
            roi_preprocessed = preprocess_input(roi_expanded)
            prediction = model.predict(roi_preprocessed)
            predicted_label_index = np.argmax(prediction, axis=1)[0]
            predicted_character = labels_dict[predicted_label_index]

            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, predicted_character, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)
        except Exception as e:
            print(f"Error during resizing or prediction: {e}")

    cv2.imshow('Image', imgOutput)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
