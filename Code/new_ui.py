import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    current_phase = "Not Setup phase"
    prev_wrist_left_y = None
    prev_wrist_right_y = None
    top_backswing_detected = False

    HIP_NEAR_THRESHOLD = 0.05  # Tolerance for wrist position relative to hips
    MIN_MOVEMENT_THRESHOLD = 0.03  # Allow some small movement

    phase_images = {}
    phase_log = []

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
            wrist_left_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
            wrist_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
            hip_left_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
            hip_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
            shoulder_left_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            shoulder_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

            hip_y_avg = (hip_left_y + hip_right_y) / 2

            # Detect Setup Phase
            if (abs(wrist_left_y - hip_y_avg) < HIP_NEAR_THRESHOLD and abs(wrist_right_y - hip_y_avg) < HIP_NEAR_THRESHOLD):
                if prev_wrist_left_y is not None and prev_wrist_right_y is not None:
                    if (abs(wrist_left_y - prev_wrist_left_y) < MIN_MOVEMENT_THRESHOLD and abs(wrist_right_y - prev_wrist_right_y) < MIN_MOVEMENT_THRESHOLD):
                        current_phase = "Setup phase"
                    else:
                        current_phase = "Setup phase"
                else:
                    current_phase = "Setup phase"

            # Detect Top Backswing Phase
            elif (wrist_left_y < shoulder_left_y and wrist_right_y < shoulder_right_y):
                current_phase = "Top backswing phase"
                top_backswing_detected = True

            # Detect Ball Impact Phase
            elif (top_backswing_detected and abs(wrist_left_y - hip_y_avg) < HIP_NEAR_THRESHOLD and abs(wrist_right_y - hip_y_avg) < HIP_NEAR_THRESHOLD):
                current_phase = "Ball impact phase"

            else:
                current_phase = ""

            prev_wrist_left_y = wrist_left_y
            prev_wrist_right_y = wrist_right_y

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ret, buffer = cv2.imencode('.jpg', frame_bgr)
            frame_jpg = buffer.tobytes()

            if current_phase and current_phase not in phase_images:
                phase_images[current_phase] = frame_jpg

            if current_phase:
                phase_log.append((frame_no, current_phase))

            yield frame_jpg, current_phase

    cap.release()
    pose.close()
    yield phase_images, phase_log

# Add custom CSS to reduce title size and center the video
st.markdown("""
    <style>
    .title {
        font-size: 25px;
        font-weight: bold;
    }
    .centered-video {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .phase-placeholder {
        border: 2px solid black; 
        padding: 10px; 
        text-align: center; 
        font-size: 25px; 
        border-radius: 5px; 
        height: 50px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Golf Swing Phase Detection</div><hr>", unsafe_allow_html=True)
st.write("Upload a video to detect different phases of a golf swing.")

video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    tfile_path = tfile.name

    st.write("Processing video...")

    phase_placeholder = st.empty()
    video_placeholder = st.empty()

    phase_images = {}
    phase_log = []

    for result, current_phase in process_video(tfile_path):
        if current_phase is not None:
            phase_placeholder.markdown(
                f"<div class='phase-placeholder'>{current_phase}</div>",
                unsafe_allow_html=True
            )
            video_placeholder.image(result, channels='BGR')
            time.sleep(0.03)  # Adjust delay to control frame rate
        else:
            phase_images, phase_log = result

    phase_placeholder.empty()
    video_placeholder.empty()

    st.write("Detected Phases:")
    columns = st.columns(3)
    col_idx = 0
    for phase, image in phase_images.items():
        if phase:  # Only show phases that are detected
            with columns[col_idx]:
                st.image(image, channels='BGR')
                st.markdown(f"### {phase}")
            col_idx = (col_idx + 1) % 3

    st.write("Phase Log:")
    for log in phase_log:
        st.write(f"Frame {log[0]}: {log[1]}")
