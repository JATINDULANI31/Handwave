import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
import time
import pyttsx3



model = joblib.load('Training/Models/rf.h5')
# Initialize mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to stop webcam
def stop_webcam_callback():
    st.session_state['use_webcam'] = False

# Function to reset sentence
def reset_sentence_callback():
    st.session_state['sentence'] = ""
    
# Function to reset current sentence
def reset_current_sentence_callback():
    st.session_state['current_sentence'] = ""

def reset_session_state_callback():
    st.session_state['sentence'] = ""
    st.session_state['current_sentence'] = ""
    
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Streamlit app
# Custom CSS for centering sidebar buttons and styling text
st.markdown("""
    <style>
    .centered-buttons {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        gap: 10px;
        padding: 20px;
    }
    .fancy-text {
        font-size: 1.5em;
        color: #ff6347;
        text-shadow: 2px 2px 4px #000000;
    }
    .prediction-text {
        font-size: 1.2em;
        color: #32cd32;
        text-shadow: 1px 1px 2px #000000;
    }
    .stButton button {
        width: 100%;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Sign Language to Text</h1>", unsafe_allow_html=True)
st.set_option('deprecation.showfileUploaderEncoding', False)

# Sidebar options
if 'use_webcam' not in st.session_state:
    st.session_state['use_webcam'] = False
if 'sentence' not in st.session_state:
    st.session_state['sentence'] = ""
    
# Initialize session state for current sentence
if 'current_sentence' not in st.session_state:
    st.session_state['current_sentence'] = ""

st.sidebar.markdown('---')
    
# Custom sidebar buttons using st.markdown
st.sidebar.markdown('<div class="centered-buttons">', unsafe_allow_html=True)
if st.sidebar.button('Use Webcam'):
    st.session_state['use_webcam'] = True
if st.sidebar.button("Stop Webcam"):
    stop_webcam_callback()
if st.sidebar.button("New Sentence"):
    reset_sentence_callback()
st.sidebar.markdown('</div>', unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)
st.sidebar.markdown('---')
st.markdown("<h3 style='text-align: center; color: #0000FF;'>Output</h3>", unsafe_allow_html=True)

stframe = st.empty()  # Ensure stframe is defined before the loop
prediction_text = st.empty()  # Placeholder for prediction text
sentence_text = st.empty()  # Placeholder for sentence

if st.session_state['use_webcam']:
    # Display current sentence
    st.sidebar.markdown(f'**Current Sentence:** {st.session_state["current_sentence"]}')
    # Add a button to speak the current sentence in the sidebar
    if st.session_state['current_sentence']:
        if st.sidebar.button('Speak Sentence', key='speak_sentence'):
            speak_text(st.session_state['current_sentence'])
    cap = cv2.VideoCapture(0)
    data_aux = []
    x_ = []
    y_ = []
    counter = 0
    # Initialize variables for accumulating predictions
    sentence = ""
    last_prediction = ""
    prediction_start_time = time.time()
    prediction_threshold = 2 # seconds
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

        is_running = True
        while is_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break

            H, W, _ = frame.shape

            # Process the frame here (hand detection, feature extraction, prediction)
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

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    x1 = int(min(x_) * W) - 20
                    y1 = int(min(y_) * H) - 20
                    x2 = int(max(x_) * W) + 20
                    y2 = int(max(y_) * H) + 20
                    
                    prediction = model.predict([np.asarray(data_aux)])
                    # Draw rectangle around the hand
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, str(prediction), (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (256,256, 0), 3,
                    cv2.LINE_AA)
                    
                    prediction_text.markdown(f'<div class="prediction-text">Prediction: {prediction}</div>', unsafe_allow_html=True)

                    # Check if the prediction is stable for the threshold time
                    current_time = time.time()
                    if prediction == last_prediction:
                        if current_time - prediction_start_time > prediction_threshold:
                            # st.session_state['current_sentence'] += prediction
                            st.session_state['sentence'] += prediction
                            prediction_start_time = current_time
                    else:
                        last_prediction = prediction
                        prediction_start_time = current_time

                    # Update sentence display
                    sentence_text.markdown(f'<div class="fancy-text">Sentence: {st.session_state["sentence"]}</div>', unsafe_allow_html=True)
                    st.session_state['current_sentence'] = st.session_state["sentence"]
                    curr_sentence = st.session_state["sentence"]
                    # Reset auxiliary data
                    data_aux.clear()
                    x_.clear()
                    y_.clear()
            # Convert frame to byte array and update Streamlit frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_image = buffer.tobytes()
            stframe.image(frame_image, channels='BGR', use_column_width=True)
            
            # Debugging: Print counter or frame size
            # print(f"Counter: {counter}, Frame size: {len(frame_image)}")

            counter += 1  # Increment counter

            # Stop running if 'Stop Webcam' is clicked
            if not st.session_state['use_webcam']:
                is_running = False

            if cv2.waitKey(25) & 0xFF == 27:  # Press 'ESC' to exit
                break
    cap.release()
    cv2.destroyAllWindows()

# Ensure the webcam is released if the app is stopped
if not st.session_state['use_webcam'] and 'cap' in locals() and cap.isOpened():
    cap.release()
