"""
capture_images.py
-----------------
Interactive webcam script to collect face images for one person at a time.

Usage:
    python src/data_collection/capture_images.py --person "Alice" --count 50

Controls (while capture window is open):
    SPACE  – capture a frame
    a      – auto-capture mode (captures every N frames automatically)
    q      – quit / finish for this person
"""

import argparse
import os
import time

import cv2

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def parse_args():
    parser = argparse.ArgumentParser(description="Capture face images from webcam.")
    parser.add_argument("--person", required=True, help="Name of the person (used as folder name)")
    parser.add_argument("--count", type=int, default=50, help="Target number of images to capture")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument(
        "--auto-interval",
        type=float,
        default=0.5,
        help="Seconds between auto-captures when in auto mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Prepare output directory
    person_dir = os.path.join(RAW_DATA_DIR, args.person)
    os.makedirs(person_dir, exist_ok=True)

    # Find next available image index
    existing = [
        f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".png"))
    ]
    img_index = len(existing)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    # Load face detector for live preview
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    auto_mode = False
    last_auto_time = 0.0
    captured = 0

    print(f"\n[INFO] Capturing images for: {args.person}")
    print(f"[INFO] Target: {args.count} images  |  Save dir: {person_dir}")
    print("[INFO] Controls: SPACE=capture  a=toggle auto  q=quit\n")

    while captured < args.count:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame.")
            continue

        display = frame.copy()

        # Draw face detection previews
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for x, y, w, h in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # HUD
        mode_str = "AUTO" if auto_mode else "MANUAL"
        cv2.putText(
            display,
            f"{args.person} | {captured}/{args.count} | {mode_str}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2,
        )

        cv2.imshow("Capture – press SPACE, a, or q", display)

        key = cv2.waitKey(1) & 0xFF

        should_capture = False
        if key == ord(" "):
            should_capture = True
        elif key == ord("a"):
            auto_mode = not auto_mode
            print(f"[INFO] Auto mode {'ON' if auto_mode else 'OFF'}")
        elif key == ord("q"):
            print("[INFO] Quit requested.")
            break

        if auto_mode and (time.time() - last_auto_time) >= args.auto_interval:
            should_capture = True
            last_auto_time = time.time()

        if should_capture:
            filename = os.path.join(person_dir, f"{args.person}_{img_index:04d}.jpg")
            cv2.imwrite(filename, frame)
            img_index += 1
            captured += 1
            print(f"[SAVED] {filename}  ({captured}/{args.count})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Captured {captured} images for '{args.person}' in {person_dir}")


if __name__ == "__main__":
    main()
