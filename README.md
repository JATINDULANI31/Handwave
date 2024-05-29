<h1><b>Handwave</b></h1>

<h2><b>Overview</b></h2>
This project aims to detect sign language gestures using the <b>MediaPipe</b> library and a trained <b>Random Forest (RF)</b> model. The detected signs are then converted into text for better accessibility.

## Features
- Real-time sign detection using webcam or video input.
- Preprocessing of hand landmarks using MediaPipe.
- Classification of hand gestures using the trained RF model.
- Text output corresponding to the detected sign.

## Requirements
- Python 3.7+
- OpenCV
- MediaPipe
- scikit-learn (for RF model)
- NumPy
- Download the requirements.txt file for better experience.

## Installation
1. Clone this repository: <b>git clone https://github.com/JATINDULANI31/Handwave.git</b>
2. Install the required packages: Using <b>requirements.txt</b> file.

## Usage
1. Run the main script: Enter <b>streamlit run main.py</b> in your terminal. Make sure that you are in the desired path of the folder that is you are in the folder path.
2. Position your hand in front of the webcam or provide a video input.
3. The detected sign will be displayed on the screen, along with the corresponding text.

## Data Collection 
1. Collect a dataset of hand gestures in sign language. Each sample should include hand landmarks (extracted using MediaPipe) and the corresponding label (text).<br>
(For references u can see the <b>Data Collect</b> folder.)

## Training the ML Model
1. Preprocess the hand landmarks (normalize, flatten, etc.).
2. Train an RF classifier using scikit-learn.
3. Save the trained model for inference.<br>
(For references u can see the <b>sign_detection.ipynb</b> file in the Training Folder.)

## Acknowledgments
- MediaPipe
- Scikit-learn
- OpenCV
- Streamlit

