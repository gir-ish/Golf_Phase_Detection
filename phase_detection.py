import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle_between_vectors(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return np.degrees(angle)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    output_dir = tempfile.mkdtemp()

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

    MIN_MOVEMENT_THRESHOLD = 0.01
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
                    if (abs(wrist_left_y - prev_wrist_left_y) < MIN_MOVEMENT_THRESHOLD and abs(wrist_right_y - prev_wrist_right_y) < MIN_MOVEMENT_THRESHOLD):
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
            elif (abs(wrist_left_y - mid_swing_y) < MID_SWING_THRESHOLD and abs(wrist_right_y - mid_swing_y) < MID_SWING_THRESHOLD and not top_backswing_detected and not ball_impact_detected):
                current_phase = "Mid backswing phase"
            elif (wrist_left_y < shoulder_left_y and wrist_right_y < shoulder_right_y and not mid_downswing_detected and not ball_impact_detected):
                current_phase = "Top backswing phase"
                top_backswing_detected = True
                top_backswing_frame = frame_no
            elif (abs(wrist_left_y - mid_swing_y) < MID_SWING_THRESHOLD and abs(wrist_right_y - mid_swing_y) < MID_SWING_THRESHOLD and top_backswing_detected and frame_no > top_backswing_frame):
                current_phase = "Mid downswing phase"
                mid_downswing_detected = True
                mid_downswing_frame = frame_no
            elif (wrist_left_y < shoulder_left_y and wrist_right_y < shoulder_right_y and ball_impact_detected and frame_no > ball_impact_frame):
                current_phase = "Follow through phase"
            else:
                current_phase = ""

            prev_wrist_left_y = wrist_left_y
            prev_wrist_right_y = wrist_right_y

            cv2.putText(frame, f"Phase: {current_phase}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Save the frame for each detected phase
            if current_phase and current_phase not in saved_phases:
                phase_filename = os.path.join(output_dir, f"{current_phase.replace(' ', '_')}.png")
                cv2.imwrite(phase_filename, frame)
                saved_phases.add(current_phase)

        cv2.imshow("Pose Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    return output_dir

st.title("Golf Swing Phase Detection")
st.write("Upload a video to detect different phases of a golf swing.")

video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    tfile_path = tfile.name

    st.write("Processing video...")
    output_dir = process_video(tfile_path)

    st.write("Detected phases saved to:", output_dir)
    st.write("Example frames from detected phases:")
    
    for phase_image in os.listdir(output_dir):
        st.image(os.path.join(output_dir, phase_image), caption=phase_image)
