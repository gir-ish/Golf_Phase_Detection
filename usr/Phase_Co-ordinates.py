import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Video file path
video_path = r"C:\Users\HP\Downloads\GOLF\Video\3.mp4"

# Check if the file exists
if not os.path.exists(video_path):
    print(f"Error: The file at {video_path} does not exist.")
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}.")
    exit()

output_dir = r"C:\Users\HP\Downloads\GOLF\snapshot\video2"
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, r"C:\Users\HP\Downloads\pose_coordinates3.csv")

with open(csv_path, mode='w', newline='') as csv_file:
    fieldnames = ['frame_no', 'phase']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

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

    MIN_MOVEMENT_THRESHOLD = 0.05
    HIP_NEAR_THRESHOLD = 0.05
    MID_SWING_THRESHOLD = 0.05

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
            wrist_left = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            wrist_right = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            hip_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            shoulder_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            shoulder_right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            hip_y_avg = (hip_left.y + hip_right.y) / 2
            shoulder_y_avg = (shoulder_left.y + shoulder_right.y) / 2
            mid_swing_y = (shoulder_y_avg + hip_y_avg) / 2

            # Ensure the current phase persists for a few more milliseconds if it's Ball Impact
            if ball_impact_detected and frame_no <= ball_impact_frame + BALL_IMPACT_DURATION:
                current_phase = "Ball impact phase"
            elif (abs(wrist_left.y - hip_y_avg) < HIP_NEAR_THRESHOLD and abs(wrist_right.y - hip_y_avg) < HIP_NEAR_THRESHOLD):
                if prev_wrist_left_y is not None and prev_wrist_right_y is not None:
                    if (abs(wrist_left.y - prev_wrist_left_y) < MIN_MOVEMENT_THRESHOLD and abs(wrist_right.y - prev_wrist_right_y) < MIN_MOVEMENT_THRESHOLD):
                        if mid_downswing_detected and frame_no > mid_downswing_frame:
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
            elif (abs(wrist_left.y - mid_swing_y) < MID_SWING_THRESHOLD and abs(wrist_right.y - mid_swing_y) < MID_SWING_THRESHOLD and not top_backswing_detected and not ball_impact_detected):
                current_phase = "Mid backswing phase"
            elif (wrist_left.y < shoulder_left.y and wrist_right.y < shoulder_right.y and not mid_downswing_detected and not ball_impact_detected):
                current_phase = "Top backswing phase"
                top_backswing_detected = True
                top_backswing_frame = frame_no
            elif (abs(wrist_left.y - mid_swing_y) < MID_SWING_THRESHOLD and abs(wrist_right.y - mid_swing_y) < MID_SWING_THRESHOLD and top_backswing_detected and frame_no > top_backswing_frame):
                current_phase = "Mid downswing phase"
                mid_downswing_detected = True
                mid_downswing_frame = frame_no
            elif (wrist_left.y < shoulder_left.y and wrist_right.y < shoulder_right.y and ball_impact_detected and frame_no > ball_impact_frame):
                current_phase = "Follow through phase"
            else:
                current_phase = ""

            prev_wrist_left_y = wrist_left.y
            prev_wrist_right_y = wrist_right.y

            print(f"Frame: {frame_no}, hip_y_avg: {hip_y_avg:.4f}, shoulder_y_avg: {shoulder_y_avg:.4f}, mid_swing_y: {mid_swing_y:.4f}")
            print(f"wrist_left_y: {wrist_left.y:.4f}, wrist_right_y: {wrist_right.y:.4f}")
            print(f"Detected phase: {current_phase}")

            cv2.putText(frame, f"Phase: {current_phase}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Save the frame for each detected phase
            if current_phase and current_phase not in saved_phases:
                phase_filename = os.path.join(output_dir, f"{current_phase.replace(' ', '_')}.png")
                cv2.imwrite(phase_filename, frame)
                saved_phases.add(current_phase)

                # Write the frame number and phase to CSV
                writer.writerow({
                    'frame_no': frame_no,
                    'phase': current_phase
                })

        cv2.imshow("Pose Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
pose.close()