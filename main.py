import streamlit as st
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Function to extract frames from video
def extract_frames(video_path, output_folder, frame_rate=1):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_list = []
    while success:
        if count % frame_rate == 0:
            frame_path = os.path.join(output_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_path, image)
            frame_list.append(frame_path)
        success, image = vidcap.read()
        count += 1

    vidcap.release()
    return frame_list

# Function to detect deepfake on a single frame
def detect_deepfake_on_frame(model, frame_path):
    img = image.load_img(frame_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    prediction = model.predict(img_array)
    return prediction[0][0]

# Function to aggregate predictions across frames
def aggregate_video_results(predictions):
    avg_prediction = np.mean(predictions)
    return "Deepfake" if avg_prediction > 0.5 else "Real"

# Load pre-trained model
@st.cache_resource
def load_meso_model():
    return load_model('mesonet_model.h5')  

# Custom CSS to enhance the look
st.markdown("""
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
    }
    .sub-title {
        font-size: 24px;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
    }
    .result-text {
        font-size: 28px;
        font-weight: bold;
        color: #F57C00;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app
st.markdown('<h1 class="title">Deepfake Video Detection</h1>', unsafe_allow_html=True)

# File uploader section with columns
st.markdown('<h3 class="sub-title">Upload a video for deepfake detection</h3>', unsafe_allow_html=True)
uploaded_video = st.file_uploader("", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    video_path = f"temp_{uploaded_video.name}"
    
    # Save the uploaded video to a temporary file
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    # Display the uploaded video
    st.video(uploaded_video)

    # Create columns for better layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.write("### Extracting frames from video...")
    
    with col2:
        # Progress bar
        progress_text = st.empty()
        progress_bar = st.progress(0)

        # Extract frames from the video
        frames_folder = 'extracted_frames'
        frames = extract_frames(video_path, frames_folder, frame_rate=30)  # Extract every 30th frame
        progress_bar.progress(50)
        progress_text.text(f"Extracted {len(frames)} frames")

    # Load the model
    model = load_meso_model()

    # Predict deepfake likelihood for each frame
    st.write("### Analyzing frames for deepfake...")
    predictions = []
    
    for i, frame_path in enumerate(frames):
        pred = detect_deepfake_on_frame(model, frame_path)
        predictions.append(pred)
        progress_bar.progress(50 + int(50 * (i + 1) / len(frames)))  # Update progress

    # Aggregate results
    final_result = aggregate_video_results(predictions)
    st.markdown(f'<h2 class="result-text">Final Verdict: {final_result}</h2>', unsafe_allow_html=True)

    # Cleanup: remove saved video and frames
    os.remove(video_path)
    for frame_path in frames:
        os.remove(frame_path)
    os.rmdir(frames_folder)
else:
    st.info("Please upload a video to start the detection process.")
