import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Video file path
video_path = r"C:\Users\giris\Downloads\RESEARCH\Golf_suretec\Video\Video\3.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
    
horizontal_line_length = 500 
# Create a new directory to save phase images
output_dir = "golf_swing_phases"
os.makedirs(output_dir, exist_ok=True)

# CSV file path
csv_file_path = os.path.join(output_dir, "golf_swing_angles.csv")
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Line1", "Line2", "Angle_A", "Angle_B", "Angle_C"])

front_shoulder = None

def calculate_angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return np.degrees(angle)

def get_line_equation(p1, p2):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    slope = delta_y / delta_x
    intercept = p1[1] - slope * p1[0]
    return slope, intercept

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

        # Get the required landmarks for line calculation
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z * w])
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z * w])

        if front_shoulder is None:
            right_shoulder_depth = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z
            left_shoulder_depth = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z

            # Determine which shoulder is closer to the camera
            if right_shoulder_depth < left_shoulder_depth:
                front_shoulder = 'right'
            else:
                front_shoulder = 'left'

        if front_shoulder == 'right':
            shoulder_to_highlight = right_shoulder
            rear_shoulder = left_shoulder
        else:
            shoulder_to_highlight = left_shoulder
            rear_shoulder = right_shoulder

        # Define a horizontal line from the highlighted shoulder
        if front_shoulder == 'right':
            horizontal_line_start = (shoulder_to_highlight[0], shoulder_to_highlight[1], shoulder_to_highlight[2])
            horizontal_line_end = (shoulder_to_highlight[0] + 200, rear_shoulder[1], shoulder_to_highlight[2])  # Adjust 200 as needed
        else:
            horizontal_line_start = (shoulder_to_highlight[0], shoulder_to_highlight[1], shoulder_to_highlight[2])
            horizontal_line_end = (shoulder_to_highlight[0] - 200, rear_shoulder[1], shoulder_to_highlight[2])  # Adjust 200 as needed

        # Convert horizontal line end to numpy array
        horizontal_line_end = np.array(horizontal_line_end)

        # Draw the horizontal line from the highlighted shoulder
        cv2.line(frame, (int(horizontal_line_start[0]), int(horizontal_line_start[1])), (int(horizontal_line_end[0]), int(horizontal_line_end[1])), (255, 0, 0), 2)  # Blue line

        # Draw the line between left shoulder and right shoulder
        cv2.line(frame, (int(left_shoulder[0]), int(left_shoulder[1])), (int(right_shoulder[0]), int(right_shoulder[1])), (0, 0, 255), 2)  # Red line

        # Highlight the front shoulder
        cv2.circle(frame, (int(shoulder_to_highlight[0]), int(shoulder_to_highlight[1])), 10, (0, 255, 0), -1)  # Green circle

        # Calculate vectors for angle calculation
        vector_ab = rear_shoulder - shoulder_to_highlight
        vector_ac = horizontal_line_end - shoulder_to_highlight
        vector_bc = horizontal_line_end - rear_shoulder

        # Calculate angles at A, B, and C
        angle_a = calculate_angle(vector_ab, vector_ac)
        angle_b = calculate_angle(-vector_ab, vector_bc)
        angle_c = calculate_angle(-vector_ac, -vector_bc)

        # Create a strip for displaying angles
        strip_height = 50
        strip = np.zeros((strip_height, w, 3), dtype=np.uint8)

        # Annotate angles on the strip
        cv2.putText(strip, f"A: {angle_a:.2f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(strip, f"B: {angle_b:.2f} deg", (w//3 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(strip, f"C: {angle_c:.2f} deg", (2*w//3 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Combine the frame and the strip
        combined_frame = np.vstack((strip, frame))

        cv2.putText(combined_frame, f"Frame: {frame_no}", (10, strip_height + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw and label points A, B, and C
        points = [shoulder_to_highlight, rear_shoulder, horizontal_line_end]
        labels = ['A', 'B', 'C']
        for idx, point in enumerate(points):
            cv2.circle(combined_frame, (int(point[0]), int(point[1]) + strip_height), 5, (0, 0, 255), -1)
            cv2.putText(combined_frame, labels[idx], (int(point[0]) + 10, int(point[1]) + strip_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # Calculate line equations
        line1_slope, line1_intercept = get_line_equation(left_shoulder, right_shoulder)
        line2_slope, line2_intercept = get_line_equation(shoulder_to_highlight, horizontal_line_end)

        # Annotate line equations on the video
        cv2.putText(combined_frame, f"Line1: y = {line1_slope:.2f}x + {line1_intercept:.2f}", (10, strip_height + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(combined_frame, f"Line2: y = {line2_slope:.2f}x + {line2_intercept:.2f}", (10, strip_height + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Save the combined frame
        phase_filename = f"{output_dir}/frame_{frame_no}.png"
        cv2.imwrite(phase_filename, combined_frame)

        # Append data to CSV
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([frame_no, f"y={line1_slope:.2f}x+{line1_intercept:.2f}", f"y={line2_slope:.2f}x+{line2_intercept:.2f}", angle_a, angle_b, angle_c])

        cv2.imshow("Pose Estimation", combined_frame)

    else:
        print(f"No pose landmarks detected in frame {frame_no}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
