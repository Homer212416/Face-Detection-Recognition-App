import cv2
import argparse
import os
import time

"""
always to run the script at the project root directory !!!
```bash
python src/data_collection/capture_images.py --person "Alice" --count 50
```
"""

def parse_args():
    parser = argparse.ArgumentParser(
                                    prog='capture_images',
                                    description='to capture images for training',
                                    )
    parser.add_argument('--person', type=str, required=True)
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--auto-interval', type=float, default=0.5)

    return parser.parse_args()


def main():

    # ---------------------------- set up before the capture loop -------------------------------

    args = parse_args()
    path = os.path.join('data', 'raw', args.person)
    os.makedirs(path, exist_ok=True)

    # to locate the idx to start with
    img_index = len(os.listdir(path))

    # open the webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened(): 
        raise RuntimeError(f'Camera index was {args.camera} tried, but failed.')

    # load Haar Cascade detector
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    

    # state variables for the loop
    auto_mode = False
    should_capture = False
    last_capture_time = time.time()

    # ---------------------------------- capture loop --------------------------------------------
    while img_index < args.count:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            display = frame.copy() # this is to display to the programmers, the picture with rectangles
            frame = cv2.cvtColor(frame, cv2.COLOR)



if __name__ == "__main__":
    main()