import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time
import base64

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def calculate_angle_between_vectors(v1, v2):
    """
    Calculates the angle between two vectors in degrees.
    """
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip for numerical stability
    return np.degrees(angle)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Define the order of phases
    phases_order = [
        "Setup Phase",
        "Mid Backswing Phase",
        "Top Backswing Phase",
        "Mid Downswing Phase",
        "Ball Impact Phase",
        "Follow Through Phase"
    ]

    # Initialize a dictionary to store the first detected frame for each phase
    phase_images = {phase: None for phase in phases_order}

    # Initialize variables for phase detection
    current_phase = "Not Setup Phase"
    prev_wrist_left_y = None
    prev_wrist_right_y = None
    top_backswing_detected = False
    mid_downswing_detected = False
    ball_impact_detected = False
    top_backswing_frame = -2
    mid_downswing_frame = -2
    ball_impact_frame = -2

    mid_backswing_wrist_left_y = None
    mid_backswing_wrist_right_y = None

    BALL_IMPACT_DURATION = 2  # Duration in frames to display Ball Impact phase

    MIN_MOVEMENT_THRESHOLD = 0.01
    HIP_NEAR_THRESHOLD = 0.05
    MID_SWING_THRESHOLD = 0.05

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        if result.pose_landmarks:
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

            # Phase Detection Logic
            if ball_impact_detected and frame_no > ball_impact_frame + BALL_IMPACT_DURATION:
                current_phase = "Follow Through Phase"
            elif abs(wrist_left_y - mid_swing_y) < MID_SWING_THRESHOLD and \
                 abs(wrist_right_y - mid_swing_y) < MID_SWING_THRESHOLD and \
                 not top_backswing_detected and not ball_impact_detected:
                current_phase = "Mid Backswing Phase"
                mid_backswing_wrist_left_y = wrist_left_y
                mid_backswing_wrist_right_y = wrist_right_y
            elif wrist_left_y < shoulder_left_y and wrist_right_y < shoulder_right_y and \
                 not mid_downswing_detected and not ball_impact_detected:
                current_phase = "Top Backswing Phase"
                top_backswing_detected = True
                top_backswing_frame = frame_no
            elif mid_backswing_wrist_left_y is not None and mid_backswing_wrist_right_y is not None and \
                 abs(wrist_left_y - mid_backswing_wrist_left_y) < MID_SWING_THRESHOLD and \
                 abs(wrist_right_y - mid_backswing_wrist_right_y) < MID_SWING_THRESHOLD and \
                 top_backswing_detected and frame_no > top_backswing_frame:
                current_phase = "Mid Downswing Phase"
                mid_downswing_detected = True
                mid_downswing_frame = frame_no
            elif abs(wrist_left_y - hip_y_avg) < HIP_NEAR_THRESHOLD and \
                 abs(wrist_right_y - hip_y_avg) < HIP_NEAR_THRESHOLD:
                if prev_wrist_left_y is not None and prev_wrist_right_y is not None:
                    if abs(wrist_left_y - prev_wrist_left_y) < MIN_MOVEMENT_THRESHOLD and \
                       abs(wrist_right_y - prev_wrist_right_y) < MIN_MOVEMENT_THRESHOLD:
                        if mid_downswing_detected and frame_no > mid_downswing_frame:
                            current_phase = "Ball Impact Phase"
                            ball_impact_detected = True
                            ball_impact_frame = frame_no
                        else:
                            current_phase = "Setup Phase"
                        top_backswing_detected = False
                        mid_downswing_detected = False
                    else:
                        current_phase = ""
                else:
                    if mid_downswing_detected and frame_no > mid_downswing_frame:
                        current_phase = "Ball Impact Phase"
                        ball_impact_detected = True
                        ball_impact_frame = frame_no
                    else:
                        current_phase = "Setup Phase"
                    top_backswing_detected = False
                    mid_downswing_detected = False
            else:
                current_phase = ""

            prev_wrist_left_y = wrist_left_y
            prev_wrist_right_y = wrist_right_y

            # **Removed Overlay of Phase Information on Video Frames**

            # Encode frame for display
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ret, buffer = cv2.imencode('.jpg', frame_bgr)
            frame_jpg = buffer.tobytes()

            # Store the first detected frame for each phase
            if current_phase in phases_order and phase_images[current_phase] is None:
                phase_images[current_phase] = frame_jpg

        processed_frames += 1
        yield frame_jpg, current_phase, processed_frames, total_frames

    cap.release()
    pose.close()
    yield phase_images, None, processed_frames, total_frames

# Streamlit UI Configuration

# Custom CSS for styling
st.markdown("""
    <style>
    /* General Styles */
    body {
        background-color: #f5f5f5;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .phase-placeholder {
        border: 2px solid #2E86C1; 
        background-color: #D6EAF8;
        padding: 15px; 
        text-align: center; 
        font-size: 24px; 
        border-radius: 10px; 
        height: 60px;
        margin-bottom: 20px;
        color: #1B4F72;
    }
    .detected-phases {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 30px; /* Increased gap for more space between phase cards */
    }
    .phase-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        background-color: #fff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        width: 250px;
        height: 300px; /* Fixed height to accommodate image and text */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .phase-card img {
        width: 200px;   /* Fixed width */
        height: 200px;  /* Fixed height */
        object-fit: cover; /* Ensures image covers the area without distortion */
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #aaa;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown("<div class='title'>🏌️‍♂️ Golf Swing Phase Detection</div><hr>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a golf swing video to automatically detect and visualize its different phases.</div>", unsafe_allow_html=True)

# File Uploader
video_file = st.file_uploader("🎥 **Upload Your Golf Swing Video**", type=["mp4", "avi", "mov", "mkv"])

if video_file is not None:
    with st.spinner("🔍 Processing video..."):
        try:
            # Save uploaded video to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            tfile_path = tfile.name

            phase_placeholder = st.empty()
            video_placeholder = st.empty()
            progress_bar = st.progress(0)
            progress_text = st.empty()

            phase_images = {}
            detected_phases = []

            for result, current_phase, processed, total in process_video(tfile_path):
                if current_phase is not None:
                    # Update phase placeholder
                    phase_placeholder.markdown(
                        f"<div class='phase-placeholder'>{current_phase}</div>",
                        unsafe_allow_html=True
                    )
                    # Display video frame
                    video_placeholder.image(result, channels='BGR', use_column_width=True)
                    # Update progress bar
                    progress = processed / total if total > 0 else 0
                    progress_bar.progress(progress)
                    progress_text.text(f"Processing frame {processed} of {total}")
                    # Collect detected phases
                    if current_phase not in detected_phases and current_phase != "Not Setup Phase":
                        detected_phases.append(current_phase)
                    time.sleep(0.01)  # Adjust delay as needed for smoother progress
                else:
                    phase_images = result

            # Cleanup placeholders
            phase_placeholder.empty()
            video_placeholder.empty()
            progress_bar.empty()
            progress_text.empty()

            st.success("✅ Video processing complete!")

            # Define the order of phases for fixed grid positions
            phases_order = [
                "Setup Phase",
                "Mid Backswing Phase",
                "Top Backswing Phase",
                "Mid Downswing Phase",
                "Ball Impact Phase",
                "Follow Through Phase"
            ]

            # Display Detected Phases in Fixed Grid Layout
            st.markdown("### 🏆 Detected Phases:")
            if phases_order:
                st.markdown("<div class='detected-phases'>", unsafe_allow_html=True)
                for phase in phases_order:
                    image = phase_images.get(phase)
                    if image:
                        # Proper Base64 encoding
                        encoded_image = base64.b64encode(image).decode('utf-8')
                        st.markdown(f"""
                            <div class='phase-card'>
                                <img src="data:image/jpeg;base64,{encoded_image}" alt="{phase}">
                                <h3>{phase}</h3>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Placeholder for missing phases
                        st.markdown(f"""
                            <div class='phase-card'>
                                <img src="https://via.placeholder.com/200x200.png?text=No+Image" alt="{phase}">
                                <h3>{phase}</h3>
                                <p style="color: #aaa; font-size: 14px;">Not Detected</p>
                            </div>
                        """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("⚠️ No distinct phases detected.")

        except Exception as e:
            st.error(f"An error occurred during video processing: {e}")

    # Footer
    st.markdown("<div class='footer'>Developed by Your Name | © 2024 Golf Swing Analyzer</div>", unsafe_allow_html=True)
