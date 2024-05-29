import streamlit as st
import PIL
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp
import joblib
import time
import pyttsx3
import io
from io import BytesIO
import streamlit_lottie
from streamlit_lottie import st_lottie
import requests
print(cv2.__version__)

model = joblib.load('knn.h5')
# Function to load Lottie animation from URL
def load_lottie_url(url: str):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

# URL of the Lottie animation JSON
lottie_url = "https://lottie.host/c0601bde-ef3c-4223-8488-e1f1696bf1c8/j5pLi1B7t9.json"
lottie_animation = load_lottie_url(lottie_url)

st.sidebar.markdown('<h3 style="text-align: center; color: grey; font-size: 25px;">Choose the App Mode</h3>', unsafe_allow_html=True)
app_mode = st.sidebar.selectbox('',
['About App','Sign Language to Text']
)
# Define your other functions and constants here

if app_mode =='About App':
    # Add custom CSS for styling the text with animation
    st.write("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Croissant+One&display=swap');
        .big-font {
            font-size: 90px !important;
            font-family: 'Croissant One', cursive;
            color: #FF9633;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

    # Layout for the title and animation
    col1, col2 = st.columns([0.8, 0.2])

    # Add content to the first column
    with col1:
        st.write('<p class="big-font">HandWave</p>', unsafe_allow_html=True)

    # Add content to the second column
    with col2:
        # Display the Lottie animation
        if lottie_animation:
            with st.container():
                # st.markdown("<div class=''>", unsafe_allow_html=True)
                st_lottie(lottie_animation, height=140, width=140)
                # st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.write("Failed to load the Lottie animation.")

    # Insert a horizontal line
    st.write("<hr style='border: 2px solid #FF9633;'>", unsafe_allow_html=True)

    # Embed custom CSS for styling
    st.write("""
    <style>
    .heading-color {
        font-family: 'Open Sans', sans-serif;
        font-size: 30px;
        color: #3366ff;
    }

    .paragraph-text {
        font-family: 'Roboto', sans-serif;
        font-size: 18px;
        color: #555555;
    }
    </style>
    """, unsafe_allow_html=True)

    # Display the content
    st.write("<div>", unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: blue; font-size: 30px;">Overview</h3>', unsafe_allow_html=True)
    st.write("<p class='paragraph-text' style='text-align: center;'>Sign language is a crucial mode of communication for individuals with hearing impairments. This application aims to bridge the communication gap by translating sign language gestures into text. By leveraging computer vision and machine learning techniques, the application can recognize and interpret various sign language gestures in real-time.</p>", unsafe_allow_html=True)
    st.write("</div>", unsafe_allow_html=True)

    st.write("<div>", unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: blue; font-size: 30px;">Model Details</h3>', unsafe_allow_html=True)
    st.write("<p class='paragraph-text' style='text-align: center;'>The MediaPipe library is utilized for hand detection, which identifies and tracks the position of hands in the video stream. Feature extraction techniques are applied to extract relevant information from the detected hand gestures. The model is trained using various machine learning models, which learns to predict the corresponding text based on the detected hand gestures. The training data consists of a diverse set of sign language gestures having a training size of around 77k predictions, enabling the model to recognize a wide range of expressions accurately.</p>", unsafe_allow_html=True)
    st.write("</div>", unsafe_allow_html=True)

    st.write("<div>", unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: blue; font-size: 30px;">Hand Gestures</h3>', unsafe_allow_html=True)
    # Load your images
    image1 = Image.open('Hand Sign Alphabet.jpeg')
    image2 = Image.open('combined_image1.jpg')
    image3 = Image.open('combined_image2.jpg')

    # Create a list of images
    images = [image1, image2, image3]
    # captions = ["Image 1", "Image 2", "Image 3"]

        # Initialize session state for the image index
    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0

        # Create layout with image and navigation buttons
    col1, col2, col3 = st.columns([0.5,5,0.5])  # 1/4 for each button, 1/2 for image

    # Left arrow button
    with col1:
        for i in range(18):
            st.markdown(' ') #Used to create some space between the filter widget and the comments section
        if st.button("🔙"):
            st.session_state.image_index = (st.session_state.image_index - 1) % len(images)

    # Display the current image
    with col2:
        st.image(images[st.session_state.image_index])

    # Right arrow button
    with col3:
        for i in range(18):
            st.markdown(' ') #Used to create some space between the filter widget and the comments section
        if st.button("🔜"):
            st.session_state.image_index = (st.session_state.image_index + 1) % len(images)
    st.write("</div>", unsafe_allow_html=True)

    st.write("<div>", unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: blue; font-size: 30px;">Try the Model</h3>', unsafe_allow_html=True)
    st.write("<p class='paragraph-text' style='text-align: center;'>Use the sidebar to navigate to the Sign Language Recognition page and try out the model in action!</p>", unsafe_allow_html=True)
    st.write("</div>", unsafe_allow_html=True)
    for i in range(5):
        st.sidebar.markdown(' ') #Used to create some space between the filter widget and the comments section
    st.sidebar.markdown('<h3 style="text-align: center; color: grey; font-size: 20px;">Check out the whole project implementation in my GitHub repo.</h3>', unsafe_allow_html=True)
    st.sidebar.markdown("""
        <div style='text-align: center;'>
            <a href="https://github.com/JATINDULANI31/Handwave" target="_blank">
                <button style="padding: 10px 20px; background-color: black; color: white; border: none; border-radius: 5px; cursor: pointer;">Go to GitHub</button>
            </a>
        </div>
    """, unsafe_allow_html=True)
    for i in range(5):
        st.sidebar.markdown(' ') #Used to create some space between the filter widget and the comments section
    st.sidebar.markdown('<h1 style="text-align: center;">FEEDBACK</h1>', unsafe_allow_html=True)
    with st.sidebar.form(key='columns_in_form',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
        rating=st.slider("Please rate the app", min_value=1, max_value=5, value=1,help='Drag the slider to rate the app. This is a 1-5 rating scale where 5 is the highest rating')
        text=st.text_input(label='Please leave your feedback here')
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write('Thanks for your feedback!')
            st.markdown('Your Rating:')
            st.markdown(rating)
            st.markdown('Your Feedback:')
            st.markdown(text) 
elif app_mode == 'Sign Language to Text':
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
                        cv2.putText(frame, str(prediction), (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 3,
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
    

