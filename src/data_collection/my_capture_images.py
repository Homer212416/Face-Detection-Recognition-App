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
    parser.add_argument('--interval', type=float, default=0.5)

    return parser.parse_args()


def main():

    # ---------------------------- set up before the capture loop -------------------------------

    args = parse_args()
    directory_path = os.path.join('data', 'raw', args.person)
    os.makedirs(directory_path, exist_ok=True)

    # to locate the idx to start with
    img_index = len(os.listdir(directory_path))

    # open the webcam
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened(): 
        raise RuntimeError(f'Camera {args.camera} was tried, but failed.')

    # load Haar Cascade detector, it is used to make the rectangle on display
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # the timer that drives auto-capture — every time the elapsed time since           
    # last_capture_time exceeds auto_interval, a frame is saved and last_capture_time 
    # is reset to time.time() to start the next interval.
    last_capture_time = time.time() 

    # ---------------------------------- capture loop --------------------------------------------
    while img_index < args.count:
        ret, frame = camera.read() # take a picture
        if not ret:
            break
        else:
            display = frame.copy() # this is to display to the users, the pictures with rectangles
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # only one face in our image; -> left-up coordinates (x,y) and width, height
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0: continue
            x,y,w,h = faces[0] 
            cv2.rectangle(display, (x,y), (x+w, y+h), (0,255,0), 3)
            # HUD
            cv2.putText(display, f"{args.person}  {img_index}/{args.count}")
            # show
            cv2.imshow("Capture", display)
            cv2.waitkey(1) # waitKey is equivalent to flush here
            # auto-capture timing, it stores one frame when time lapses
            if (time.time() - last_capture_time) >= args.interval: 
                last_capture_time = time.time()
                # store the frame
                file_name = f'{args.person}_{img_index:04d}.jpg'
                file_path = os.path.join(directory_path,file_name)
                cv2.imwrite(file_path, frame)
                img_index += 1
    camera.release()
    cv2.destroyAllWindows()

            


            


            
            



if __name__ == "__main__":
    main()