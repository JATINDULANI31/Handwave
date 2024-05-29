import os
import pickle
import csv
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = 'data'
# Open CSV file for writing
OUTPUT_DIR = 'data_in_csv'
os.makedirs(OUTPUT_DIR,exist_ok=True)
mx = 0
data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    # print(dir_)
    csv_path = os.path.join(OUTPUT_DIR, f"{dir_}.csv")
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = []
        header.append('Class')
        for i in range(21):
            header.append(f'x{i}')
            header.append(f'y{i}')
        writer.writerow(header)

    data1 = []
    k = 1
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with mp_hands.Hands(static_image_mode=True) as hands:
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks on the image
                    mp_drawing.draw_landmarks(
                        img_rgb,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    mx = max(mx, len(hand_landmarks.landmark))
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x - min(x_)
                        y = hand_landmarks.landmark[i].y - min(y_)
                        data_aux.append(x)
                        data_aux.append(y)
                #   print(dir_path)
                    data1.append(data_aux)
        print(k)
        k+=1
    print(dir_)
    # Construct CSV file path using the output directory
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data1:
            l = []
            l.append(dir_)
            for cell in row:
                l.append(cell)
            writer.writerow(l)
        labels.append(dir_)

print(labels)
print("Data saved to data1.csv")
