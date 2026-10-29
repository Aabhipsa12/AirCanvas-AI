"""
AirCanvas AI - Milestone 2: Hand Landmark Detection
Uses MediaPipe Tasks API to detect hand landmarks in real time from the webcam feed,
tracks the index finger tip (Landmark 8), and displays its pixel coordinates.
"""

import os
import sys
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
    HandLandmarksConnections,
    drawing_utils,
)


def open_camera():
    """
    Attempts to open the default webcam.
    Tries Windows DirectShow (cv2.CAP_DSHOW) first for lower latency,
    and falls back to OpenCV's default backend if needed.
    """
    print("Attempting to open webcam with DirectShow (cv2.CAP_DSHOW)...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("DirectShow failed. Falling back to default backend...")
        cap = cv2.VideoCapture(0)

    return cap


def create_hand_landmarker(model_path):
    """
    Initializes and returns a MediaPipe HandLandmarker detector configured for video stream.
    """
    if not os.path.isfile(model_path):
        print(f"\n[ERROR] Model file not found at: {model_path}")
        print("Please ensure 'models/hand_landmarker.task' exists.")
        sys.exit(1)

    # Configure HandLandmarker options:
    # 1. BaseOptions points to the trained model weights file
    # 2. RunningMode.VIDEO optimizes detection across consecutive frames using temporal tracking
    # 3. num_hands=1 restricts detection to a single hand for simplicity and performance
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return HandLandmarker.create_from_options(options)


def main():
    # Locate model path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "hand_landmarker.task")

    # Step 1: Open webcam
    cap = open_camera()
    if not cap.isOpened():
        print("\n[ERROR] Could not open webcam.")
        print("Please check camera connections, privacy permissions, and make sure no other app is using it.")
        sys.exit(1)

    # Step 2: Initialize HandLandmarker
    print("Loading MediaPipe HandLandmarker model...")
    landmarker = create_hand_landmarker(model_path)
    print("HandLandmarker loaded successfully!")
    print("\nPress 'q' or 'Q' while the 'AirCanvas AI' window is active to exit.\n")

    last_timestamp_ms = 0

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                print("\n[ERROR] Failed to read frame from webcam. Stream ended.")
                break

            # 1. Mirror the frame horizontally (1 = flip around Y-axis)
            # This makes hand movement intuitive, like looking into a mirror.
            mirrored_frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = mirrored_frame.shape

            # 2. Color Space Conversion: BGR -> RGB
            # OpenCV reads images in BGR format (Blue, Green, Red).
            # MediaPipe's neural networks are trained on and require RGB (Red, Green, Blue).
            rgb_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)

            # Wrap the numpy RGB array into a MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # 3. Monotonically increasing timestamp (required by RunningMode.VIDEO)
            # MediaPipe uses timestamps to correlate hand positions across frames.
            current_timestamp_ms = int(time.time() * 1000)
            if current_timestamp_ms <= last_timestamp_ms:
                current_timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = current_timestamp_ms

            # 4. Run hand detection on the current video frame
            detection_result = landmarker.detect_for_video(mp_image, current_timestamp_ms)

            # 5. Check if any hand was detected
            if detection_result.hand_landmarks:
                # We configured num_hands=1, so we take the first detected hand
                first_hand_landmarks = detection_result.hand_landmarks[0]

                # Draw all 21 landmarks and the connecting skeleton lines
                drawing_utils.draw_landmarks(
                    mirrored_frame,
                    first_hand_landmarks,
                    HandLandmarksConnections.HAND_CONNECTIONS,
                )

                # 6. Locate Landmark 8: Index Finger Tip
                # Normalized coordinates range from 0.0 (top-left) to 1.0 (bottom-right).
                index_finger_tip = first_hand_landmarks[8]

                # 7. Convert Normalized Coordinates -> Pixel Coordinates
                # Multiplying normalized values by frame width and height gives the exact pixel location.
                pixel_x = int(index_finger_tip.x * frame_width)
                pixel_y = int(index_finger_tip.y * frame_height)

                # 8. Highlight the Index Fingertip
                # Draw a prominent filled circle (magenta) with a white outline
                cv2.circle(mirrored_frame, (pixel_x, pixel_y), 12, (255, 0, 255), cv2.FILLED)
                cv2.circle(mirrored_frame, (pixel_x, pixel_y), 14, (255, 255, 255), 2)

                # 9. Display pixel coordinates on screen
                coord_text = f"Index Tip (Landmark 8): ({pixel_x}, {pixel_y})"
                cv2.putText(
                    mirrored_frame,
                    coord_text,
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                # Inform the user when no hand is in the frame
                cv2.putText(
                    mirrored_frame,
                    "Hand: Not Detected",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Display the resulting frame
            cv2.imshow("AirCanvas AI - Hand Tracking", mirrored_frame)

            # Listen for exit key ('q' or 'Q')
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Exit requested. Closing...")
                break

    finally:
        # Step 3: Guaranteed Cleanup
        print("Releasing resources...")
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("Webcam and MediaPipe resources released successfully.")


if __name__ == "__main__":
    main()
