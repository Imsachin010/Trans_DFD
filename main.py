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

# Streamlit app
st.title("Deepfake Video Detection")

uploaded_video = st.file_uploader("Upload a video for deepfake detection", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    
    video_path = f"temp_{uploaded_video.name}"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    st.video(uploaded_video)

    # Extract frames from the video
    st.write("Extracting frames from video...")
    frames_folder = 'extracted_frames'
    frames = extract_frames(video_path, frames_folder, frame_rate=30)  # Extract every 30th frame
    st.write(f"Extracted {len(frames)} frames")

    # Load the model
    model = load_meso_model()

    # Predict deepfake likelihood for each frame
    st.write("Analyzing frames for deepfake...")
    predictions = []
    for frame_path in frames:
        pred = detect_deepfake_on_frame(model, frame_path)
        predictions.append(pred)

    # Aggregate results
    final_result = aggregate_video_results(predictions)
    st.write(f"Final verdict: **{final_result}**")

    # Cleanup: remove saved video and frames
    os.remove(video_path)
    for frame_path in frames:
        os.remove(frame_path)
    os.rmdir(frames_folder)
