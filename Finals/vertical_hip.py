import cv2
import mediapipe as mp
import numpy as np
import os
import json

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Video file path
video_path = r"/home/girish/GIT/Video/3.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create a new directory to save phase images
output_dir = r"C:\Users\giris\Downloads\golf_swing_phases"
os.makedirs(output_dir, exist_ok=True)

# JSON file path
json_file_path = os.path.join(output_dir, r"C:\Users\giris\Downloads\hip_coordinates.json")
data = []

# Variables to store the front hip point in the first frame
initial_hip_point = None
front_hip_index = None
line_shift = 20  # Amount to shift the line

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)
    h, w, c = frame.shape

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        # Determine front hip point (right or left)
        right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h])
        left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h])

        if initial_hip_point is None:
            # Determine which hip is closer to the camera
            right_hip_depth = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z
            left_hip_depth = landmarks[mp_pose.PoseLandmark.LEFT_HIP].z
            if right_hip_depth < left_hip_depth:
                initial_hip_point = right_hip
                front_hip_index = mp_pose.PoseLandmark.RIGHT_HIP
            else:
                initial_hip_point = left_hip
                front_hip_index = mp_pose.PoseLandmark.LEFT_HIP

        # Get the current front hip point
        current_hip_point = np.array([landmarks[front_hip_index].x * w, landmarks[front_hip_index].y * h])

        # Shift the line slightly left for left hip (point 24) and right for right hip (point 23)
        if front_hip_index == mp_pose.PoseLandmark.RIGHT_HIP:
            line_x = current_hip_point[0] - line_shift  # Shift right
        else:
            line_x = current_hip_point[0] + line_shift  # Shift left

        # Draw a vertical line from the current hip point (moving with hip)
        cv2.line(frame, (int(line_x), 0), (int(line_x), h), (0, 0, 255), 1)  # Red line

        # Highlight the current hip point
        cv2.circle(frame, (int(current_hip_point[0]), int(current_hip_point[1])), 10, (0, 255, 0), -1)  # Green circle

        # Create a strip for displaying frame number
        strip_height = 50
        strip = np.zeros((strip_height, w, 3), dtype=np.uint8)

        # Annotate frame number on the strip
        cv2.putText(strip, f"Frame: {frame_no}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Combine the frame and the strip
        combined_frame = np.vstack((strip, frame))

        # Save the combined frame
        phase_filename = f"{output_dir}/frame_{frame_no}.png"
        cv2.imwrite(phase_filename, combined_frame)

        # Save coordinates to JSON
        data.append({
            "frame": frame_no,
            "hip_x": float(current_hip_point[0]),
            "hip_y": float(current_hip_point[1])
        })

        cv2.imshow("Pose Estimation", combined_frame)

    else:
        print(f"No pose landmarks detected in frame {frame_no}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()

# Save the data to a JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f"Hip coordinates have been saved to {json_file_path}")
