import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Video file path
video_path = r"C:\Users\giris\Downloads\RESEARCH\Golf_suretec\Video\Video\2.mp4"

# Check if the file exists
if not os.path.exists(video_path):
    print(f"Error: The file at {video_path} does not exist.")
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}.")
    exit()

output_dir = r"C:\Users\giris\Downloads\RESEARCH\Golf_suretec\video2"
os.makedirs(output_dir, exist_ok=True)

current_phase = "Not Setup phase"
prev_wrist_left_y = None
prev_wrist_right_y = None
top_backswing_detected = False
mid_downswing_detected = False
ball_impact_detected = False
top_backswing_frame = -2
mid_downswing_frame = -2
ball_impact_frame = -2

BALL_IMPACT_DURATION = 2  # Duration in frames to display Ball Impact phase

# Thresholds for phase detection - change these values as needed
MIN_MOVEMENT_THRESHOLD = 0.5  # Minimal movement threshold for wrist detection
HIP_NEAR_THRESHOLD = 0.05      # Threshold for wrist and hip proximity to detect Setup phase
MID_SWING_THRESHOLD = 0.2      # Threshold for wrist and mid-swing proximity to detect Mid backswing and Mid downswing phases
FAST_MOVEMENT_THRESHOLD = 0.2  # Threshold for detecting fast movements

saved_phases = set()

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
        shoulder_y_avg = (shoulder_left_y + shoulder_right_y) / 2
        mid_swing_y = (shoulder_y_avg + hip_y_avg) / 2

        # Ensure the current phase persists for a few more milliseconds if it's Ball Impact
        if ball_impact_detected and frame_no <= ball_impact_frame + BALL_IMPACT_DURATION:
            current_phase = "Ball impact phase"
        elif (abs(wrist_left_y - hip_y_avg) < HIP_NEAR_THRESHOLD and abs(wrist_right_y - hip_y_avg) < HIP_NEAR_THRESHOLD):
            if prev_wrist_left_y is not None and prev_wrist_right_y is not None:
                movement_speed_left = abs(wrist_left_y - prev_wrist_left_y)
                movement_speed_right = abs(wrist_right_y - prev_wrist_right_y)
                if (movement_speed_left < MIN_MOVEMENT_THRESHOLD and movement_speed_right < MIN_MOVEMENT_THRESHOLD):
                    if (movement_speed_left > FAST_MOVEMENT_THRESHOLD or movement_speed_right > FAST_MOVEMENT_THRESHOLD):
                        current_phase = "Fast movement"
                    elif mid_downswing_detected and frame_no > mid_downswing_frame:
                        current_phase = "Ball impact phase"
                        ball_impact_detected = True
                        ball_impact_frame = frame_no
                    else:
                        current_phase = "Setup phase"
                    top_backswing_detected = False
                    mid_downswing_detected = False
                else:
                    current_phase = ""
            else:
                if mid_downswing_detected and frame_no > mid_downswing_frame:
                    current_phase = "Ball impact phase"
                    ball_impact_detected = True
                    ball_impact_frame = frame_no
                else:
                    current_phase = "Setup phase"
                top_backswing_detected = False
                mid_downswing_detected = False
        elif (abs(wrist_left_y - hip_y_avg) < MID_SWING_THRESHOLD and abs(wrist_right_y - hip_y_avg) < MID_SWING_THRESHOLD and not top_backswing_detected and not ball_impact_detected and current_phase == "Setup phase"):
            current_phase = "Mid backswing phase"
        elif (wrist_left_y < shoulder_left_y and wrist_right_y < shoulder_right_y and not mid_downswing_detected and not ball_impact_detected):
            current_phase = "Top backswing phase"
            top_backswing_detected = True
            top_backswing_frame = frame_no
        elif (abs(wrist_left_y - hip_y_avg) < MID_SWING_THRESHOLD and abs(wrist_right_y - hip_y_avg) < MID_SWING_THRESHOLD and top_backswing_detected and frame_no > top_backswing_frame):
            current_phase = "Mid downswing phase"
            mid_downswing_detected = True
            mid_downswing_frame = frame_no
        elif (wrist_left_y < shoulder_left_y and wrist_right_y < shoulder_right_y and ball_impact_detected and frame_no > ball_impact_frame):
            current_phase = "Follow through phase"
        else:
            current_phase = ""

        prev_wrist_left_y = wrist_left_y
        prev_wrist_right_y = wrist_right_y

        print(f"Frame: {frame_no}, hip_y_avg: {hip_y_avg:.4f}, shoulder_y_avg: {shoulder_y_avg:.4f}, mid_swing_y: {mid_swing_y:.4f}")
        print(f"wrist_left_y: {wrist_left_y:.4f}, wrist_right_y: {wrist_right_y:.4f}")
        print(f"Detected phase: {current_phase}")

        cv2.putText(frame, f"Phase: {current_phase}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Save every frame with the current phase in the filename
        phase_filename = os.path.join(output_dir, f"{current_phase.replace(' ', '_')}_frame_{frame_no}.png")
        cv2.imwrite(phase_filename, frame)
        saved_phases.add(current_phase)

    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
