
import cv2
import mediapipe as mp
import numpy as np
import os
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

video_path = r"C:\Users\giris\Downloads\RESEARCH\Golf_suretec\Video\Video\3.mp4"
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
output_dir = "golf_swing_phases"
os.makedirs(output_dir, exist_ok=True)
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

        
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h])
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h])
        left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h])
        right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h])

        
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2
        
        hip_midpoint = (left_hip + right_hip) / 2

        
        vertical_end_point = np.array([hip_midpoint[0], shoulder_midpoint[1]])

        
        cv2.line(frame, (int(shoulder_midpoint[0]), int(shoulder_midpoint[1])), (int(hip_midpoint[0]), int(hip_midpoint[1])), (255, 0, 0), 2)  # Red line
        cv2.line(frame, (int(hip_midpoint[0]), int(hip_midpoint[1])), (int(vertical_end_point[0]), int(vertical_end_point[1])), (0, 255, 0), 2)  # Green vertical line

        
        cv2.circle(frame, (int(shoulder_midpoint[0]), int(shoulder_midpoint[1])), 5, (0, 0, 255), -1)  # Point 1
        cv2.putText(frame, '1', (int(shoulder_midpoint[0]) + 10, int(shoulder_midpoint[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.circle(frame, (int(hip_midpoint[0]), int(hip_midpoint[1])), 5, (0, 0, 255), -1)  # Point 2
        cv2.putText(frame, '2', (int(hip_midpoint[0]) + 10, int(hip_midpoint[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.circle(frame, (int(vertical_end_point[0]), int(vertical_end_point[1])), 5, (0, 0, 255), -1)  # Point 3
        cv2.putText(frame, '3', (int(vertical_end_point[0]) + 10, int(vertical_end_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        
        vector_shoulder_to_hip = shoulder_midpoint - hip_midpoint
        vector_hip_to_vertical = vertical_end_point - hip_midpoint
        angle_between_lines = calculate_angle_between_vectors(vector_shoulder_to_hip, vector_hip_to_vertical)

        cv2.putText(frame, f"Angle: {angle_between_lines:.2f} degrees", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        print(f"Frame: {frame_no}, Angle: {angle_between_lines:.2f} degrees")

        
        phase_filename = f"{output_dir}/frame_{frame_no}.png"
        cv2.imwrite(phase_filename, frame)

    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
pose.close()