"""
AirCanvas AI - Milestone 1: Webcam Feed Test
Demonstrates accessing the webcam, mirroring the feed, and clean teardown using OpenCV.
"""

import sys
import cv2


def open_camera():
    """
    Attempts to open the default webcam.
    First tries DirectShow (cv2.CAP_DSHOW) which is faster and more reliable on Windows.
    Falls back to the default backend if DirectShow fails.
    """
    print("Attempting to open webcam using DirectShow (cv2.CAP_DSHOW)...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("DirectShow failed. Retrying with default backend...")
        cap = cv2.VideoCapture(0)

    return cap


def main():
    cap = open_camera()

    # Step 1: Check if the camera initialized properly
    if not cap.isOpened():
        print("\n[ERROR] Could not open webcam.")
        print("Possible causes:")
        print("  1. Another application is using the camera (e.g., Zoom, Teams, browser).")
        print("  2. Windows Camera privacy settings are blocking access.")
        print("  3. No camera device is available at index 0.")
        sys.exit(1)

    print("\nWebcam opened successfully!")
    print("Press 'q' or 'Q' while the 'AirCanvas AI' window is active to exit.\n")

    # Step 2: Continuous capture loop wrapped in try/finally for guaranteed cleanup
    try:
        while True:
            # Read a single frame from the camera
            # 'success' is a boolean flag (True if frame captured)
            # 'frame' is the raw BGR image as a NumPy array
            success, frame = cap.read()

            if not success or frame is None:
                print("\n[ERROR] Failed to read frame from webcam. Stream interrupted.")
                break

            # Flip the frame horizontally (1 = horizontal flip across Y axis)
            # This creates a natural mirror effect so your hand moves in the expected direction
            mirrored_frame = cv2.flip(frame, 1)

            # Display the processed frame in a window titled "AirCanvas AI"
            cv2.imshow("AirCanvas AI", mirrored_frame)

            # cv2.waitKey(1) waits 1 millisecond for a key event
            # '& 0xFF' masks the value to get the standard 8-bit ASCII code
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Exit key pressed. Closing...")
                break

    finally:
        # Step 3: Always release the hardware and destroy GUI windows
        print("Releasing camera and closing windows...")
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("Cleanup completed successfully.")


if __name__ == "__main__":
    main()
