import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import time

# Constants
DEMO_IMAGE = 'media\stand.jpg'
DEMO_VIDEO = 'media\run.mp4'  # Replace with your default demo video if available

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

width, height = 368, 368

# Load Pose Estimation Model
@st.cache_resource
def load_model():
    try:
        model_path = "graph_opt.pb"  # Ensure this file exists in the working directory
        net = cv2.dnn.readNetFromTensorflow(model_path)
        return net
    except cv2.error as e:
        st.error("Error loading the TensorFlow model. Make sure 'graph_opt.pb' exists.")
        return None

net = load_model()

# Title
st.title("📌 Human Pose Estimation using OpenCV")
st.markdown("Upload an **image** or **video** to estimate human pose keypoints.")

# Input Selection
input_type = st.radio("Choose Input Type:", ("Image", "Video"))

# File Upload
if input_type == "Image":
    img_file_buffer = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
elif input_type == "Video":
    video_file_buffer = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])

# Set Threshold
thres = st.slider("Confidence Threshold", 0, 100, 20, 5) / 100

# Pose Estimation Function
def pose_detector(frame):
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (width, height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()[:, :19, :, :]

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x, y = (frameWidth * point[0]) / out.shape[3], (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thres else None)

    for pair in POSE_PAIRS:
        partFrom, partTo = pair[0], pair[1]
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    return frame

# Process Image Input
if input_type == "Image":
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
    else:
        image = np.array(Image.open(DEMO_IMAGE))

    st.subheader("📌 Original Image")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    output_image = pose_detector(image)
    st.subheader("🔍 Pose Estimation Output")
    st.image(output_image, caption="Pose Estimated", use_column_width=True)

# Process Video Input
elif input_type == "Video":
    if video_file_buffer is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file_buffer.read())
        video_path = tfile.name
    else:
        video_path = DEMO_VIDEO

    st.subheader("📹 Video Preview")
    st.video(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file.")

    frame_placeholder = st.empty()
    st.subheader("📍 Processing Video...")

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(temp_output, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output_frame = pose_detector(frame)
        out.write(output_frame)

        # Convert frame for display
        frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        time.sleep(1 / fps)

    cap.release()
    out.release()

    st.subheader("🎬 Processed Video Output")
    st.video(temp_output)
    
# Footer
st.markdown("**Developed using OpenCV & Streamlit**")
