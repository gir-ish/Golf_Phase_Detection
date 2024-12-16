import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True)
mp_drawing = mp.solutions.drawing_utils

# Video file path
video_path = r'C:\Users\giris\Downloads\RESEARCH\Golf_suretec\test1.mp4'


# Open video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and get pose landmarks
    result = pose.process(image_rgb)

    # Create a blank white image
    h, w, c = frame.shape
    opImg = np.ones([h, w, c], dtype=np.uint8) * 255

    if result.pose_landmarks:
        # Draw pose landmarks on the original frame
        mp_drawing.draw_landmarks(
            frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
        )

        # Draw pose landmarks on the blank image
        mp_drawing.draw_landmarks(
            opImg, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
        )

    # Concatenate the original frame and the blank image
    concatenated_image = np.concatenate((frame, opImg), axis=1)

    # Display the concatenated image
    cv2.imshow("Pose Estimation", concatenated_image)

    # Press 'q' to exit the video display window
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()
