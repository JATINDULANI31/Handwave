import os
import cv2

# Define data directory and labels
DATA_DIR = './data'
l = "a"
dataset_size = 2500

# Create data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
# Create subdirectories for each label
for char in range(ord('a'), ord('z')+1):
    label_dir = os.path.join(DATA_DIR, chr(char))
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

# Open the webcam (ensure the correct index, it might be 0, 1, or 2 depending on your setup)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise ValueError("Unable to open webcam. Check if the webcam is connected and the correct index is used.")

# Collect data for each label
for char in range(ord('a'), ord('z')+1):
    print(f'Collecting data for class {chr(char)}')
    
    # Wait for user to be ready
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from webcam. Retrying...")
            continue

        cv2.putText(frame, 'Ready? Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Capture dataset_size number of images for the current label
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from webcam. Retrying...")
            continue

        # Display the frame
        cv2.imshow('frame', frame)
        cv2.waitKey(30)

        # Save the frame to the appropriate directory
        filename = os.path.join(DATA_DIR, chr(char), f'{counter}.jpg')
        cv2.imwrite(filename, frame)
        
        counter += 1

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
