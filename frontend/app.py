import streamlit as st
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp
import joblib
import time
import pyttsx3
from io import BytesIO
from streamlit_lottie import st_lottie
import requests

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
            <a href="https://github.com/JATINDULANI31" target="_blank">
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
    # model = load_model()
    
    # Add your model execution code here
    
    # Example placeholder code
    st.write("Placeholder: Model execution code goes here")
    

