import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input


# Function to get the bounding box from hand landmarks
def get_bounding_box(hand_landmarks, img_width, img_height, padding=20):
    x_coords = [landmark.x * img_width for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y * img_height for landmark in hand_landmarks.landmark]

    x_min = max(0, int(min(x_coords) - padding))
    x_max = min(img_width, int(max(x_coords) + padding))
    y_min = max(0, int(min(y_coords) - padding))
    y_max = min(img_height, int(max(y_coords) + padding))

    return x_min, y_min, x_max, y_max

# Load the VGG16 model
model = load_model('sign_detection_model1.h5')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Dictionary mapping label index to sign language characters
labels_dict = {
    0: 'I love you', 1: 'No', 2: 'Please', 3: 'Thank you', 4: 'Yes', 5: 'a', 6: 'b', 7: 'c', 8: 'd', 9: 'e', 10: 'f',
    11: 'g', 12: 'h', 13: 'i', 14: 'j', 15: 'k', 16: 'l', 17: 'm', 18: 'n', 19: 'o', 20: 'p', 21: 'q', 22: 'r', 23: 's',
    24: 't', 25: 'u', 26: 'v', 27: 'w', 28: 'x', 29: 'y', 30: 'z'
}

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            x_ = [landmark.x for landmark in hand_landmarks.landmark]
            y_ = [landmark.y for landmark in hand_landmarks.landmark]

            x1 = int(min(x_) * W) - 30
            y1 = int(min(y_) * H) - 30
            x2 = int(max(x_) * W) + 30
            y2 = int(max(y_) * H) + 0

            # Extract the region of interest (ROI)
            roi = frame_rgb[y1:y2, x1:x2]

            if roi.shape[0] == 0 or roi.shape[1] == 0:
                continue

            # Resize and preprocess the ROI for the model
            roi_resized = cv2.resize(roi, (224, 224))
            roi_expanded = np.expand_dims(roi_resized, axis=0)
            roi_preprocessed = preprocess_input(roi_expanded)

            # Make prediction
            prediction = model.predict(roi_preprocessed)
            print(prediction)
            predicted_label_index = np.argmax(prediction, axis=1)[0]
            predicted_character = labels_dict[predicted_label_index]

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    cv2.imshow('frame', frame)
    # Display the resulting frame
    cv2.imshow('Sign Language Recognition', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
