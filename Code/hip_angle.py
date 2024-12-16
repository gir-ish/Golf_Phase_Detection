import cv2
import mediapipe as mp
import numpy as np
import os
import math
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

# Create a new directory to save phase images
output_dir = "golf_swing_phases"
os.makedirs(output_dir, exist_ok=True)

front_hip = None

# CSV file to save the results
csv_filename = "golf_swing_analysis.csv"
with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'Line1 Equation', 'Line2 Equation', 'Angle A (degrees)', 'Angle B (degrees)', 'Angle C (degrees)'])

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
            right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h])
            left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h])

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
                horizontal_line_start = (hip_to_highlight[0], hip_to_highlight[1])
                horizontal_line_end = (hip_to_highlight[0] + 200, rear_hip[1])  # Adjust 200 as needed
            else:
                horizontal_line_start = (hip_to_highlight[0], hip_to_highlight[1])
                horizontal_line_end = (hip_to_highlight[0] - 200, rear_hip[1])  # Adjust 200 as needed

            # Convert horizontal line end to numpy array
            horizontal_line_end = np.array(horizontal_line_end)

            # Draw the horizontal line from the highlighted hip
            cv2.line(frame, (int(horizontal_line_start[0]), int(horizontal_line_start[1])), (int(horizontal_line_end[0]), int(horizontal_line_end[1])), (255, 0, 0), 2)  # Blue line

            # Draw the line between left hip and right hip
            cv2.line(frame, (int(left_hip[0]), int(left_hip[1])), (int(right_hip[0]), int(right_hip[1])), (0, 0, 255), 2)  # Red line

            # Highlight the front hip
            cv2.circle(frame, (int(hip_to_highlight[0]), int(hip_to_highlight[1])), 10, (0, 255, 0), -1)  # Green circle

            # Calculate the slopes of the two lines
            slope_line1 = (rear_hip[1] - hip_to_highlight[1]) / (rear_hip[0] - hip_to_highlight[0])
            slope_line2 = (horizontal_line_end[1] - hip_to_highlight[1]) / (horizontal_line_end[0] - hip_to_highlight[0])

            # Calculate the angle between the two lines
            def angle_between_lines(m1, m2):
                # Calculate the tangent of the angle
                tan_theta = abs((m1 - m2) / (1 + m1 * m2))
                # Calculate the angle in radians
                theta_radians = math.atan(tan_theta)
                # Convert the angle to degrees
                theta_degrees = math.degrees(theta_radians)
                return theta_degrees

            # Calculate the length of the sides of the triangle
            def calculate_distance(point1, point2):
                return np.linalg.norm(point1 - point2)

            A = hip_to_highlight
            B = rear_hip
            C = horizontal_line_end

            a = calculate_distance(B, C)
            b = calculate_distance(A, C)
            c = calculate_distance(A, B)

            # Calculate angles using the law of cosines
            angle_A = math.degrees(math.acos((b**2 + c**2 - a**2) / (2 * b * c)))
            angle_B = math.degrees(math.acos((a**2 + c**2 - b**2) / (2 * a * c)))
            angle_C = 180 - angle_A - angle_B

            # Save the results in the CSV file
            line1_eq = f"y = {slope_line1:.2f}x + {hip_to_highlight[1] - slope_line1 * hip_to_highlight[0]:.2f}"
            line2_eq = f"y = {slope_line2:.2f}x + {hip_to_highlight[1] - slope_line2 * hip_to_highlight[0]:.2f}"
            csv_writer.writerow([frame_no, line1_eq, line2_eq, angle_A, angle_B, angle_C])

            # Create a strip for displaying frame number
            strip_height = 50
            strip = np.zeros((strip_height, w, 3), dtype=np.uint8)

            # Combine the frame and the strip
            combined_frame = np.vstack((strip, frame))

            cv2.putText(combined_frame, f"Frame: {frame_no}", (10, strip_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(combined_frame, f"Angle A: {angle_A:.2f} degrees", (10, strip_height + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(combined_frame, f"Angle B: {angle_B:.2f} degrees", (10, strip_height + 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(combined_frame, f"Angle C: {angle_C:.2f} degrees", (10, strip_height + 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Draw and label points A, B, and C
            points = [hip_to_highlight, rear_hip, horizontal_line_end]
            labels = ['A', 'B', 'C']
            for idx, point in enumerate(points):
                cv2.circle(combined_frame, (int(point[0]), int(point[1]) + strip_height), 5, (0, 0, 255), -1)
                cv2.putText(combined_frame, labels[idx], (int(point[0]) + 10, int(point[1]) + strip_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

            # Draw the complete triangle
            cv2.line(combined_frame, (int(points[0][0]), int(points[0][1]) + strip_height), (int(points[1][0]), int(points[1][1]) + strip_height), (0, 255, 255), 2)
            cv2.line(combined_frame, (int(points[1][0]), int(points[1][1]) + strip_height), (int(points[2][0]), int(points[2][1]) + strip_height), (0, 255, 255), 2)
            cv2.line(combined_frame, (int(points[2][0]), int(points[2][1]) + strip_height), (int(points[0][0]), int(points[0][1]) + strip_height), (0, 255, 255), 2)

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
