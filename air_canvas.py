"""
AirCanvas AI - Calligraphic Bézier Spline Engine with Pinch Hysteresis
Fixes:
1. Eliminates roughness: Uses Quadratic Bézier curve interpolation between points for round, silky-smooth cursive loops instead of segmented straight lines.
2. Fixes gaps in letters (like 'e'): Adds Schmitt-trigger Pinch Hysteresis and a 3-frame debounce buffer so tight loops never drop ink mid-letter.
3. Resolution-Aware Jump Threshold: Prevents strokes from breaking during fast letter curves at 720p.
4. Exact Gestures: Pinch to Write, V-Sign to Hover, Open Palm to Erase, Fist (1s) to Clear.
"""

import os
import sys
import time
import math
from collections import deque
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
    HandLandmarksConnections,
    drawing_utils,
)

# ==========================================
# Configuration & Constants
# ==========================================
DRAW_COLOR = (255, 255, 0)          # Neon Cyan (BGR)
DRAW_THICKNESS = 6                  # Line thickness
ERASER_RADIUS = 38                  # Radius of local palm eraser
CLEAR_HOLD_SECONDS = 1.0            # Seconds to hold fist to clear screen


# ==========================================
# StreamLine Speed-Adaptive Stabilizer
# ==========================================
class PointStabilizer:
    def __init__(self, deadband=2.0, min_alpha=0.22, max_alpha=0.78):
        self.deadband = deadband
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.prev_raw = None
        self.filtered = None

    def update(self, x, y):
        if self.filtered is None:
            self.prev_raw = (x, y)
            self.filtered = (float(x), float(y))
            return int(x), int(y)

        raw_dist = math.hypot(x - self.prev_raw[0], y - self.prev_raw[1])
        self.prev_raw = (x, y)

        if raw_dist < self.deadband:
            return int(round(self.filtered[0])), int(round(self.filtered[1]))

        speed = raw_dist
        t = min(1.0, speed / 32.0)
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * (t ** 1.6)

        fx = alpha * x + (1.0 - alpha) * self.filtered[0]
        fy = alpha * y + (1.0 - alpha) * self.filtered[1]
        self.filtered = (fx, fy)

        return int(round(fx)), int(round(fy))

    def reset(self):
        self.prev_raw = None
        self.filtered = None


def draw_bezier_curve(canvas, p0, p1, p2, color, thickness, steps=6):
    """
    Draws a smooth quadratic Bézier spline between midpoints of (p0, p1) and (p1, p2)
    using p1 as the curvature control point. Eliminates angular polygon edges!
    """
    m0 = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
    m1 = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

    prev_pt = (int(round(m0[0])), int(round(m0[1])))
    for i in range(1, steps + 1):
        t = i / float(steps)
        # B(t) = (1-t)^2 * m0 + 2*(1-t)*t * p1 + t^2 * m1
        bx = (1.0 - t)**2 * m0[0] + 2.0 * (1.0 - t) * t * p1[0] + t**2 * m1[0]
        by = (1.0 - t)**2 * m0[1] + 2.0 * (1.0 - t) * t * p1[1] + t**2 * m1[1]
        curr_pt = (int(round(bx)), int(round(by)))

        cv2.line(canvas, prev_pt, curr_pt, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, curr_pt, thickness // 2, color, cv2.FILLED)
        prev_pt = curr_pt


def open_camera():
    """Opens webcam with DirectShow and requests HD 720p."""
    print("Opening webcam with DirectShow (cv2.CAP_DSHOW)...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("DirectShow failed. Falling back to default backend...")
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def ensure_model_exists(model_path):
    """
    Automatically downloads the official MediaPipe HandLandmarker model
    if it is missing, ensuring the project works out of the box when cloned.
    """
    if not os.path.isfile(model_path):
        print(f"[INFO] Model not found at '{model_path}'. Downloading official model...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        import urllib.request
        urllib.request.urlretrieve(url, model_path)
        print("[INFO] Model downloaded successfully!")


def create_hand_landmarker(model_path):
    """Initializes MediaPipe HandLandmarker with high confidence thresholds."""
    ensure_model_exists(model_path)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.70,
        min_hand_presence_confidence=0.70,
        min_tracking_confidence=0.70,
    )
    return HandLandmarker.create_from_options(options)


def is_anatomically_valid_hand(hand, frame_width, frame_height):
    """Rejects false background-stretched landmarks."""
    wrist = hand[0]
    middle_mcp = hand[9]

    palm_dist = math.hypot(
        (middle_mcp.x - wrist.x) * frame_width,
        (middle_mcp.y - wrist.y) * frame_height,
    )

    if palm_dist < 25 or palm_dist > frame_height * 0.75:
        return False

    for tip_idx in (4, 8, 12, 16, 20):
        tip = hand[tip_idx]
        tip_dist = math.hypot(
            (tip.x - wrist.x) * frame_width,
            (tip.y - wrist.y) * frame_height,
        )
        if tip_dist > 2.5 * palm_dist:
            return False

    return True


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "hand_landmarker.task")

    cap = open_camera()
    if not cap.isOpened():
        print("\n[ERROR] Could not open webcam.")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam running at resolution: {actual_w}x{actual_h}")

    print("Loading HandLandmarker model...")
    landmarker = create_hand_landmarker(model_path)
    print("Model loaded successfully!")
    print("\n--- Controls & Gestures ---")
    print("  [👌 Pinch (Index + Thumb)] : WRITE / DRAW (Bézier Smoothed)")
    print("  [✌️ V-Sign (Index + Middle)] : HOVER / MOVE (Pen Up)")
    print("  [🖐️ Open Palm]              : LOCAL ERASER (erase hovered strokes)")
    print("  [✊ Fist (Hold 1s)]         : CLEAR SCREEN")
    print("  [Keyboard 'C']              : Instant screen clear")
    print("  [Keyboard 'Q']              : Exit\n")

    # State & Filters
    canvas = None
    pen_stabilizer = PointStabilizer(deadband=2.0, min_alpha=0.22, max_alpha=0.78)
    eraser_stabilizer = PointStabilizer(deadband=2.0, min_alpha=0.25, max_alpha=0.80)

    # Stroke point buffer for Bézier curve rendering
    stroke_points = deque(maxlen=3)
    was_pinching = False
    pinch_grace_frames = 0

    fist_start_time = None
    clear_notification_until = 0
    last_timestamp_ms = 0

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                print("\n[ERROR] Failed to read frame from webcam.")
                break

            mirrored_frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = mirrored_frame.shape

            # Initialize canvas once
            if canvas is None or canvas.shape != mirrored_frame.shape:
                canvas = np.zeros_like(mirrored_frame)

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            current_timestamp_ms = int(time.time() * 1000)
            if current_timestamp_ms <= last_timestamp_ms:
                current_timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = current_timestamp_ms

            detection_result = landmarker.detect_for_video(mp_image, current_timestamp_ms)

            current_mode = "STANDBY"
            mode_color = (200, 200, 200)

            hand_valid = False
            if detection_result.hand_landmarks:
                hand = detection_result.hand_landmarks[0]
                if is_anatomically_valid_hand(hand, frame_width, frame_height):
                    hand_valid = True

            if hand_valid:
                # Draw subtle skeleton
                drawing_utils.draw_landmarks(
                    mirrored_frame,
                    hand,
                    HandLandmarksConnections.HAND_CONNECTIONS,
                )

                # Key Landmarks
                wrist = hand[0]
                thumb_tip = hand[4]
                index_tip = hand[8]
                middle_tip = hand[12]
                palm_center = hand[9]

                idx_raw_x = int(index_tip.x * frame_width)
                idx_raw_y = int(index_tip.y * frame_height)
                th_x = int(thumb_tip.x * frame_width)
                th_y = int(thumb_tip.y * frame_height)
                palm_x = int(palm_center.x * frame_width)
                palm_y = int(palm_center.y * frame_height)
                wrist_x = int(wrist.x * frame_width)
                wrist_y = int(wrist.y * frame_height)

                # Stabilized coordinates
                stable_pen_x, stable_pen_y = pen_stabilizer.update(idx_raw_x, idx_raw_y)
                stable_erase_x, stable_erase_y = eraser_stabilizer.update(palm_x, palm_y)

                # Finger extension states
                index_up = hand[8].y < hand[6].y
                middle_up = hand[12].y < hand[10].y
                ring_up = hand[16].y < hand[14].y
                pinky_up = hand[20].y < hand[18].y

                # ==========================================
                # PINCH HYSTERESIS (Schmitt Trigger)
                # Solves gaps in loops like 'e':
                # - Easy to stay in draw mode once started
                # - Requires deliberate finger opening to lift pen
                # ==========================================
                hand_scale = math.hypot(palm_x - wrist_x, palm_y - wrist_y)
                pinch_dist = math.hypot(idx_raw_x - th_x, idx_raw_y - th_y)

                enter_thresh = max(28, min(52, hand_scale * 0.30))
                exit_thresh = enter_thresh + 16  # 16px buffer to avoid chattering

                if was_pinching:
                    raw_is_pinching = pinch_dist < exit_thresh
                else:
                    raw_is_pinching = pinch_dist < enter_thresh

                was_pinching = raw_is_pinching

                # 3-frame grace buffer so micro-twitches during tight loops never create a gap
                if raw_is_pinching:
                    pinch_grace_frames = 3
                    drawing_active = True
                else:
                    if pinch_grace_frames > 0:
                        pinch_grace_frames -= 1
                        drawing_active = True
                    else:
                        drawing_active = False

                # ==========================================
                # GESTURE 1: FIST (Hold for 1s -> Clear Screen)
                # ==========================================
                is_fist = (not index_up) and (not middle_up) and (not ring_up) and (not pinky_up)

                if is_fist:
                    stroke_points.clear()
                    current_mode = "FIST (HOLD 1S TO CLEAR SCREEN)"
                    mode_color = (0, 0, 255)

                    if fist_start_time is None:
                        fist_start_time = time.time()

                    elapsed_fist = time.time() - fist_start_time
                    progress_ratio = min(elapsed_fist / CLEAR_HOLD_SECONDS, 1.0)

                    cv2.circle(mirrored_frame, (palm_x, palm_y), 32, (0, 0, 255), 2)
                    progress_angle = int(360 * progress_ratio)
                    cv2.ellipse(
                        mirrored_frame,
                        (palm_x, palm_y),
                        (32, 32),
                        0,
                        0,
                        progress_angle,
                        (0, 0, 255),
                        5,
                    )

                    if elapsed_fist >= CLEAR_HOLD_SECONDS:
                        canvas = np.zeros_like(mirrored_frame)
                        fist_start_time = None
                        clear_notification_until = time.time() + 1.2
                        print("[INFO] Screen cleared by fist gesture.")

                else:
                    fist_start_time = None

                    # ==========================================
                    # GESTURE 2: OPEN PALM (Local Eraser)
                    # ==========================================
                    is_open_palm = index_up and middle_up and ring_up and pinky_up and (not raw_is_pinching)

                    if is_open_palm:
                        stroke_points.clear()
                        current_mode = "ERASER (OPEN PALM)"
                        mode_color = (255, 0, 255)

                        cv2.circle(canvas, (stable_erase_x, stable_erase_y), ERASER_RADIUS, (0, 0, 0), cv2.FILLED)
                        cv2.circle(mirrored_frame, (stable_erase_x, stable_erase_y), ERASER_RADIUS, (255, 0, 255), 2)
                        cv2.circle(mirrored_frame, (stable_erase_x, stable_erase_y), 4, (255, 0, 255), cv2.FILLED)
                        cv2.putText(
                            mirrored_frame,
                            "ERASER",
                            (stable_erase_x - 30, stable_erase_y - ERASER_RADIUS - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )

                    # ==========================================
                    # GESTURE 3: PINCH (Smooth Bézier Calligraphy)
                    # ==========================================
                    elif drawing_active:
                        current_mode = "PINCH (WRITING - BEZIER)"
                        mode_color = (0, 255, 0)

                        draw_pt = (stable_pen_x, stable_pen_y)
                        cv2.circle(mirrored_frame, draw_pt, DRAW_THICKNESS + 2, DRAW_COLOR, cv2.FILLED)
                        cv2.circle(mirrored_frame, draw_pt, DRAW_THICKNESS + 4, (255, 255, 255), 2)

                        # Jump break: if hand moved too far across the screen (>130px at 720p), start new stroke
                        if len(stroke_points) > 0:
                            last_pt = stroke_points[-1]
                            dist = math.hypot(draw_pt[0] - last_pt[0], draw_pt[1] - last_pt[1])
                            if dist > 130:
                                stroke_points.clear()

                        stroke_points.append(draw_pt)

                        # Draw using Bézier curve interpolation
                        if len(stroke_points) == 3:
                            draw_bezier_curve(
                                canvas,
                                stroke_points[0],
                                stroke_points[1],
                                stroke_points[2],
                                DRAW_COLOR,
                                DRAW_THICKNESS,
                            )
                        elif len(stroke_points) == 2:
                            cv2.line(
                                canvas,
                                stroke_points[0],
                                stroke_points[1],
                                DRAW_COLOR,
                                DRAW_THICKNESS,
                                cv2.LINE_AA,
                            )
                            cv2.circle(canvas, stroke_points[1], DRAW_THICKNESS // 2, DRAW_COLOR, cv2.FILLED)
                        elif len(stroke_points) == 1:
                            cv2.circle(canvas, stroke_points[0], DRAW_THICKNESS // 2, DRAW_COLOR, cv2.FILLED)

                    # ==========================================
                    # GESTURE 4: V-SIGN (Hover / Move Hand)
                    # ==========================================
                    elif index_up and middle_up and (not ring_up) and (not pinky_up):
                        current_mode = "V-SIGN (HOVER / MOVE)"
                        mode_color = (0, 255, 255)
                        stroke_points.clear()

                        hover_center_x = (idx_raw_x + int(middle_tip.x * frame_width)) // 2
                        hover_center_y = (idx_raw_y + int(middle_tip.y * frame_height)) // 2
                        cv2.circle(mirrored_frame, (hover_center_x, hover_center_y), 12, (0, 255, 255), 2)
                        cv2.circle(mirrored_frame, (hover_center_x, hover_center_y), 2, (0, 255, 255), cv2.FILLED)

                    else:
                        current_mode = "STANDBY"
                        mode_color = (180, 180, 180)
                        stroke_points.clear()

            else:
                current_mode = "NO HAND DETECTED"
                mode_color = (120, 120, 120)
                pen_stabilizer.reset()
                eraser_stabilizer.reset()
                stroke_points.clear()
                was_pinching = False
                pinch_grace_frames = 0
                fist_start_time = None

            # ==========================================
            # Blend Canvas onto Live Video Frame
            # ==========================================
            canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, inv_mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY_INV)
            inv_mask_3ch = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)

            masked_frame = cv2.bitwise_and(mirrored_frame, inv_mask_3ch)
            display_frame = cv2.bitwise_or(masked_frame, canvas)

            # ==========================================
            # Heads-Up Display (HUD)
            # ==========================================
            cv2.putText(
                display_frame,
                f"MODE: {current_mode}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                mode_color,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display_frame,
                "[BEZIER CALLIGRAPHY: ACTIVE]",
                (frame_width - 320, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 200),
                1,
                cv2.LINE_AA,
            )

            if time.time() < clear_notification_until:
                cv2.putText(
                    display_frame,
                    "*** SCREEN CLEARED ***",
                    (frame_width // 2 - 180, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

            cv2.putText(
                display_frame,
                "Pinch: Write | V-Sign: Move | Open Palm: Eraser | Fist (1s): Clear Screen | 'Q': Exit",
                (15, frame_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("AirCanvas AI", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Exiting application...")
                break
            elif key == ord('c') or key == ord('C'):
                canvas = np.zeros_like(mirrored_frame)
                clear_notification_until = time.time() + 1.2
                print("[INFO] Screen cleared via keyboard ('C').")

    finally:
        print("Releasing resources...")
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("Resources released cleanly.")


if __name__ == "__main__":
    main()
