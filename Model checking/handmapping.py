# import os
# import pickle

# import mediapipe as mp
# import cv2
# import matplotlib.pyplot as plt


# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# DATA_DIR = './data'

# data = []
# labels = []
# for dir_ in os.listdir(DATA_DIR):
#     for img_path in os.listdir(os.path.join(DATA_DIR, dir_))[:1]:
#         data_aux = []

#         x_ = []
#         y_ = []

#         img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         results = hands.process(img_rgb)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     img_rgb,
#                     hand_landmarks,
#                     mp_hands.HAND_CONNECTIONS,
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style()
#                 )
#         #         for i in range(len(hand_landmarks.landmark)):
#         #             x = hand_landmarks.landmark[i].x
#         #             y = hand_landmarks.landmark[i].y

#         #             x_.append(x)
#         #             y_.append(y)

#         #         for i in range(len(hand_landmarks.landmark)):
#         #             x = hand_landmarks.landmark[i].x
#         #             y = hand_landmarks.landmark[i].y
#         #             data_aux.append(x - min(x_))
#         #             data_aux.append(y - min(y_))

#         #     data.append(data_aux)
#         #     labels.append(dir_)
#         plt.figure()
#         plt.imshow(img_rgb)

# plt.show()

# # f = open('data.pickle', 'wb')
# # pickle.dump({'data': data, 'labels': labels}, f)
# # f.close()
import cv2
import os
import mediapipe as mp
import numpy as np

# Initialize mediapipe hands and drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define the input and output directories
DATA_DIR = './data'
# PROCESSED_DIR = './dataprocessed'
CROPPED_DIR = './datacropped1'

# Create output directories if they don't exist
# os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CROPPED_DIR, exist_ok=True)

# Function to get the bounding box from hand landmarks
def get_bounding_box(hand_landmarks, img_width, img_height, padding=20):
    x_coords = [landmark.x * img_width for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y * img_height for landmark in hand_landmarks.landmark]

    x_min = max(0, int(min(x_coords) - padding))
    x_max = min(img_width, int(max(x_coords) + padding))
    y_min = max(0, int(min(y_coords) - padding))
    y_max = min(img_height, int(max(y_coords) + padding))

    return x_min, y_min, x_max, y_max

# Process each image in the input directory
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    # print(f"Processing directory: {dir_path}")
    # p = os.path.join(PROCESSED_DIR, dir_)
    c = os.path.join(CROPPED_DIR, dir_)
    # os.makedirs(p,exist_ok=True)
    os.makedirs(c, exist_ok=True)
    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        # print(f"Processing image: {img_path}")
        img = cv2.imread(img_path)
        
        # if img is None:
        #     print(f"Warning: Unable to load image {img_path}")
        #     continue
    
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width, _ = img.shape

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
                    
                    # Get bounding box coordinates
                    x_min, y_min, x_max, y_max = get_bounding_box(hand_landmarks, img_width, img_height)

                    # Crop the hand region from the image
                    # cropped_path_o = os.path.join(c, f'cropped_o_{img_name}')
                    # hand_img = img[y_min:y_max, x_min:x_max]
                    # cv2.imwrite(cropped_path_o, hand_img)
                    hand_img_rgb = img_rgb[y_min:y_max, x_min:x_max]
                    hand_img_bgr = cv2.cvtColor(hand_img_rgb, cv2.COLOR_RGB2BGR)
                    cropped_path_c = os.path.join(c, f'cropped_c_{img_name}')
                    # Save the cropped hand image
                    cv2.imwrite(cropped_path_c, hand_img_bgr)

                # Convert the image back to BGR for saving with OpenCV
                # img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                # Define the output path for the processed image
                # processed_path = os.path.join(p, img_name)
                # Save the processed image with landmarks
                # cv2.imwrite(processed_path, img_bgr)

print("Processed images and cropped hand images saved successfully.")