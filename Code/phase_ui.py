import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_midpoint(point1, point2):
    return (point1 + point2) / 2

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    phase_images = {}
    phase_log = []
    min_wrist_y = float('inf')
    setup_phase_detected = False

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
            left_wrist = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h])
            right_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * h])

            # Calculate midpoints
            midpoint_shoulders = calculate_midpoint(left_shoulder, right_shoulder)
            midpoint_hips = calculate_midpoint(left_hip, right_hip)
            center_point = calculate_midpoint(midpoint_shoulders, midpoint_hips)
            midpoint_wrists = calculate_midpoint(left_wrist, right_wrist)

            # Define the length to extend the vertical line
            vertical_line_length = 100  # You can adjust this length as needed

            # Calculate the points for the vertical line
            point_top = np.array([center_point[0], center_point[1] - vertical_line_length])
            point_bottom = np.array([center_point[0], center_point[1] + vertical_line_length])

            # Convert points to tuples after making them integers
            point_top = tuple(point_top.astype(int))
            point_bottom = tuple(point_bottom.astype(int))
            center_point = tuple(center_point.astype(int))
            midpoint_wrists = tuple(midpoint_wrists.astype(int))

            # Draw the vertical line and the line to midpoint of wrists
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.line(frame_bgr, point_top, point_bottom, (0, 255, 0), 2)
            cv2.line(frame_bgr, center_point, midpoint_wrists, (255, 0, 0), 2)

            # Detect Setup Phase based on wrist y-coordinate minima
            current_min_wrist_y = min(left_wrist[1], right_wrist[1])
            if current_min_wrist_y < min_wrist_y:
                min_wrist_y = current_min_wrist_y
                setup_phase_detected = True

            if setup_phase_detected:
                phase_images[frame_no] = frame_bgr
                phase_log.append((frame_no, "Setup phase"))

            ret, buffer = cv2.imencode('.jpg', frame_bgr)
            frame_jpg = buffer.tobytes()

            yield frame_jpg

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

st.markdown("<div class='title'>Golf Swing Analysis</div><hr>", unsafe_allow_html=True)
st.write("Upload a video to analyze golf swing and draw lines from the center point between shoulders and hips.")

video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    tfile_path = tfile.name

    st.write("Processing video...")

    video_placeholder = st.empty()
    phase_placeholder = st.empty()

    phase_images = {}
    phase_log = []

    for result in process_video(tfile_path):
        if isinstance(result, bytes):
            video_placeholder.image(result, channels='BGR')
            time.sleep(0.03)  # Adjust delay to control frame rate
        elif isinstance(result, tuple):
            phase_images, phase_log = result

    video_placeholder.empty()

    st.write("Detected Phases:")
    columns = st.columns(3)
    col_index = 0

    for frame_no, image in phase_images.items():
        with columns[col_index]:
            st.image(image, channels='BGR')
            st.write(f"Frame: {frame_no}")
        col_index = (col_index + 1) % 3

    st.write("Phase Log:")
    for log in phase_log:
        st.write(f"Frame {log[0]}: {log[1]}")
