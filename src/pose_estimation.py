import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

# Define body parts
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

# Define pose pairs (connections between key points)
POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Model input dimensions
IN_WIDTH = 368
IN_HEIGHT = 368
THRESHOLD = 0.2

# Set paths relative to the project structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "graph_opt.pb")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Ensure model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Error: Model file '{MODEL_PATH}' not found!")

# Load the pre-trained model
net = cv2.dnn.readNetFromTensorflow(MODEL_PATH)


def pose_detector(frame):
    """Detects and draws pose keypoints on an input frame."""
    frame_height, frame_width = frame.shape[:2]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (IN_WIDTH, IN_HEIGHT),
                                       (127.5, 127.5, 127.5), swapRB=True, crop=False))

    output = net.forward()[:, :19, :, :]  # Get first 19 keypoints
    assert len(BODY_PARTS) == output.shape[1]

    points = []
    for i in range(len(BODY_PARTS)):
        heat_map = output[0, i, :, :]
        _, confidence, _, point = cv2.minMaxLoc(heat_map)

        x = int((frame_width * point[0]) / output.shape[3])
        y = int((frame_height * point[1]) / output.shape[2])

        points.append((x, y) if confidence > THRESHOLD else None)

    # Draw lines and keypoints
    for pair in POSE_PAIRS:
        part_from, part_to = pair
        id_from, id_to = BODY_PARTS[part_from], BODY_PARTS[part_to]

        if points[id_from] and points[id_to]:
            cv2.line(frame, points[id_from], points[id_to], (0, 255, 0), 3)
            cv2.circle(frame, points[id_from], 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, points[id_to], 5, (0, 0, 255), cv2.FILLED)

    return frame


def visualize_result(image):
    """Displays the processed image using Matplotlib."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Pose Estimation using OpenCV and TensorFlow")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    # Validate input image
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Error: Image file '{args.image}' not found!")

    # Load and process image
    input_image = cv2.imread(args.image)
    output_image = pose_detector(input_image)

    # Save output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "output_image.png")
    cv2.imwrite(output_path, output_image)
    print(f"Processed image saved at: {output_path}")

    # Display result
    visualize_result(output_image)


if __name__ == "__main__":
    main()
