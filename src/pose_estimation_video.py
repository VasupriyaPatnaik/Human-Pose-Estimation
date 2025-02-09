import cv2
import numpy as np

# Define body parts and pose pairs
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Load the neural network model
net = cv2.dnn.readNetFromTensorflow("C:\Users\DELL\OneDrive\Documents\Human-Pose-Estimation\models\graph_opt.pb")

# Set input image dimensions
inWidth, inHeight = 368, 368
threshold = 0.2

def pose_estimation(cap):
    """Perform pose estimation on video frames."""
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot fetch frame.")
            break
        
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frameWidth, frameHeight = frame.shape[1], frame.shape[0]

        # Prepare input for the model
        net.setInput(cv2.dnn.blobFromImage(frame, 2.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]

        points = []
        for i in range(len(BODY_PARTS)):
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = int((frameWidth * point[0]) / out.shape[3])
            y = int((frameHeight * point[1]) / out.shape[2])
            points.append((x, y) if conf > threshold else None)
        
        # Draw skeleton
        for pair in POSE_PAIRS:
            partFrom, partTo = pair
            idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
            
            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.circle(frame, points[idFrom], 3, (0, 0, 255), -1)
                cv2.circle(frame, points[idTo], 3, (0, 0, 255), -1)
        
        # Display output
        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Load and process the video
video_path = "media\run.mp4"
cap = cv2.VideoCapture(video_path)
pose_estimation(cap)
