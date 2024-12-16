import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import json
import math

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

# List to store data for JSON
data = []

front_hip = None

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

def angle_between_lines(m1, m2):
    # Calculate the tangent of the angle
    tan_theta = abs((m1 - m2) / (1 + m1 * m2))
    # Calculate the angle in radians
    theta_radians = math.atan(tan_theta)
    # Convert the angle to degrees
    theta_degrees = math.degrees(theta_radians)
    return theta_degrees

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
        right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z * w])
        left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h, landmarks[mp_pose.PoseLandmark.LEFT_HIP].z * w])

        if front_hip is None:
            right_hip_depth = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z
            left_hip_depth = landmarks[mp_pose.PoseLandmark.LEFT_HIP].z

            # Determine which hip is closer to the camera
            if right_hip_depth < left_hip_depth:
                front_hip = 'right'
            else:
                front_hip = 'left'

        if front_hip == 'right':
            hip_to_highlight = right_hip
            rear_hip = left_hip
        else:
            hip_to_highlight = left_hip
            rear_hip = right_hip

        # Define a horizontal line from the highlighted hip
        if front_hip == 'right':
            horizontal_line_start = (hip_to_highlight[0], hip_to_highlight[1], hip_to_highlight[2])
            horizontal_line_end = (hip_to_highlight[0] + 200, rear_hip[1], hip_to_highlight[2])  # Adjust 200 as needed
        else:
            horizontal_line_start = (hip_to_highlight[0], hip_to_highlight[1], hip_to_highlight[2])
            horizontal_line_end = (hip_to_highlight[0] - 200, rear_hip[1], hip_to_highlight[2])  # Adjust 200 as needed

        # Convert horizontal line end to numpy array
        horizontal_line_end = np.array(horizontal_line_end)

        # Draw the horizontal line from the highlighted hip
        cv2.line(frame, (int(horizontal_line_start[0]), int(horizontal_line_start[1])), (int(horizontal_line_end[0]), int(horizontal_line_end[1])), (255, 0, 0), 2)  # Blue line

        # Draw the line between left hip and right hip
        cv2.line(frame, (int(left_hip[0]), int(left_hip[1])), (int(right_hip[0]), int(right_hip[1])), (0, 0, 255), 2)  # Red line

        # Highlight the front hip
        cv2.circle(frame, (int(hip_to_highlight[0]), int(hip_to_highlight[1])), 10, (0, 255, 0), -1)  # Green circle

        # Calculate vectors for angle calculation
        vector_ab = rear_hip - hip_to_highlight
        vector_ac = horizontal_line_end - hip_to_highlight
        vector_bc = horizontal_line_end - rear_hip

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
        points = [hip_to_highlight, rear_hip, horizontal_line_end]
        labels = ['A', 'B', 'C']
        for idx, point in enumerate(points):
            cv2.circle(combined_frame, (int(point[0]), int(point[1]) + strip_height), 5, (0, 0, 255), -1)
            cv2.putText(combined_frame, labels[idx], (int(point[0]) + 10, int(point[1]) + strip_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # Calculate line equations
        line1_slope, line1_intercept = get_line_equation(left_hip, right_hip)
        line2_slope, line2_intercept = get_line_equation(hip_to_highlight, horizontal_line_end)

        # Calculate the angle between the two lines
        angle_between = angle_between_lines(line1_slope, line2_slope)

        # Annotate line equations on the video
        cv2.putText(combined_frame, f"Line1: y = {line1_slope:.2f}x + {line1_intercept:.2f}", (10, strip_height + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(combined_frame, f"Line2: y = {line2_slope:.2f}x + {line2_intercept:.2f}", (10, strip_height + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(combined_frame, f"Angle between lines: {angle_between:.2f} deg", (10, strip_height + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Save the combined frame
        phase_filename = f"{output_dir}/frame_{frame_no}.png"
        cv2.imwrite(phase_filename, combined_frame)

        # Append data to CSV
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([frame_no, f"y={line1_slope:.2f}x+{line1_intercept:.2f}", f"y={line2_slope:.2f}x+{line2_intercept:.2f}", angle_a, angle_b, angle_c])

        # Append data to list for JSON
        data.append({
            "frame": frame_no,
            "line1": f"y={line1_slope:.2f}x+{line1_intercept:.2f}",
            "line2": f"y={line2_slope:.2f}x+{line2_intercept:.2f}",
            "angle_A": angle_a,
            "angle_B": angle_b,
            "angle_C": angle_c,
            "angle_between_lines": angle_between
        })

        cv2.imshow("Pose Estimation", combined_frame)

    else:
        print(f"No pose landmarks detected in frame {frame_no}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()

# Save the data to a JSON file
json_filename = os.path.join(output_dir, r"C:\Users\giris\Downloads\RESEARCH\Golf_suretec\Finals\golf_swing_angles.json")
with open(json_filename, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f"Coordinates and angles have been saved to {csv_file_path} and {json_filename}")
