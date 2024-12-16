import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Video file path
video_path = r"C:\Users\giris\Downloads\RESEARCH\Golf_suretec\Video\Video\5.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create a new directory to save phase images
output_dir = "golf_swing_phases"
os.makedirs(output_dir, exist_ok=True)

# Variables to store the head centroid in the first frame
initial_centroid = None

def calculate_centroid(points):
    return np.mean(points, axis=0)

def draw_dotted_circle(image, center, radius, color, thickness, gap):
    circumference = 2 * np.pi * radius
    num_dashes = int(circumference / (2 * gap))
    for i in range(num_dashes):
        angle1 = (i * 2 * gap) / radius
        angle2 = angle1 + (gap / radius)
        start_point = (int(center[0] + radius * np.cos(angle1)), int(center[1] + radius * np.sin(angle1)))
        end_point = (int(center[0] + radius * np.cos(angle2)), int(center[1] + radius * np.sin(angle2)))
        cv2.line(image, start_point, end_point, color, thickness, cv2.LINE_AA)

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

        # Get head landmarks (points 0 to 10)
        head_landmarks = [landmarks[i] for i in range(11)]
        head_points = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in head_landmarks])

        # Calculate the centroid of the head in each frame
        current_centroid = calculate_centroid(head_points)

        if initial_centroid is None:
            # Calculate and store the initial centroid
            initial_centroid = current_centroid

        # Draw a dotted circle at the initial centroid in all frames (static reference)
        draw_dotted_circle(frame, (int(initial_centroid[0]), int(initial_centroid[1])), 10, (255, 0, 0), 2, 5)  # Dotted blue circle with increased thickness

        # Draw a circle at the current centroid in all frames (moving with head)
        cv2.circle(frame, (int(current_centroid[0]), int(current_centroid[1])), 10, (0, 0, 255), 3)  # Red circle with increased thickness

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

        cv2.imshow("Pose Estimation", combined_frame)

    else:
        print(f"No pose landmarks detected in frame {frame_no}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()