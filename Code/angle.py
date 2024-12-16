import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Video file path
video_path = r"C:\Users\giris\Downloads\RESEARCH\Golf_suretec\Video\Video\1.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

def calculate_angle_between_vectors(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return np.degrees(angle)

def is_near_straight_line(a, b, c, threshold=10):
    angle = calculate_angle_between_vectors(a - b, c - b)
    return abs(angle - 180) < threshold or angle < threshold

# Create a new directory to save phase images
output_dir = "golf_swing_phases"
os.makedirs(output_dir, exist_ok=True)

current_phase = "Not Setup phase"
prev_wrist_left_y = None
prev_wrist_right_y = None
prev_frame_no = -1

HIP_NEAR_THRESHOLD = 0.05
MIN_MOVEMENT_THRESHOLD = 0.01
STRAIGHT_LINE_THRESHOLD = 100  # Threshold for near-straight line in degrees
MOVEMENT_THRESHOLD = 0.005  # Threshold for minimal movement

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)
    h, w, c = frame.shape

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
        )

        landmarks = result.pose_landmarks.landmark
        wrist_left_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
        wrist_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
        hip_left_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
        hip_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
        shoulder_left_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        shoulder_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

        hip_y_avg = (hip_left_y + hip_right_y) / 2

        # Check for setup phase with minimal movement
        if abs(wrist_left_y - hip_y_avg) < HIP_NEAR_THRESHOLD and abs(wrist_right_y - hip_y_avg) < HIP_NEAR_THRESHOLD:
            if prev_wrist_left_y is not None and prev_wrist_right_y is not None:
                if abs(wrist_left_y - prev_wrist_left_y) < MOVEMENT_THRESHOLD and abs(wrist_right_y - prev_wrist_right_y) < MOVEMENT_THRESHOLD:
                    current_phase = "Setup phase"
                else:
                    current_phase = "Not Setup phase"
            else:
                current_phase = "Setup phase"
        else:
            current_phase = "Not Setup phase"

        if current_phase == "Setup phase":
            # Get the required landmarks for angle calculation
            keypoint_12 = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
            keypoint_14 = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y])
            keypoint_20 = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y])
            keypoint_24 = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y])

            if is_near_straight_line(keypoint_20, keypoint_14, keypoint_12, STRAIGHT_LINE_THRESHOLD):
                # Calculate the angle between the line 12-20 and 12-24
                vector_12_20 = keypoint_20 - keypoint_12
                vector_12_24 = keypoint_24 - keypoint_12
                angle_between_lines = calculate_angle_between_vectors(vector_12_20, vector_12_24)

                cv2.putText(frame, f"Angle 12-20-24: {angle_between_lines:.2f} degrees", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, "Near straight line detected", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Not a near straight line", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        prev_wrist_left_y = wrist_left_y
        prev_wrist_right_y = wrist_right_y

        print(f"Frame: {frame_no}, hip_y_avg: {hip_y_avg:.4f}")
        print(f"wrist_left_y: {wrist_left_y:.4f}, wrist_right_y: {wrist_right_y:.4f}")
        print(f"Detected phase: {current_phase}")

        cv2.putText(frame, f"Phase: {current_phase}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Save the frame if it's in the setup phase
        if current_phase == "Setup phase":
            phase_filename = f"{output_dir}/frame_{frame_no}_{current_phase.replace(' ', '_')}.png"
            cv2.imwrite(phase_filename, frame)

    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
