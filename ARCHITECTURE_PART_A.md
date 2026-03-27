# Part A — Data Pipeline

**Assigned files:** `src/data_collection/capture_images.py` · `src/preprocessing/preprocess.py`

**Goal:** Collect raw face images from a webcam and turn them into clean, labelled, augmented dataset splits ready for training.

---

## Part A — File 1: `src/data_collection/capture_images.py`

### How to run

```bash
python src/data_collection/capture_images.py --person "Alice" --count 50
```

### File structure

Organise the file into two functions — `parse_args()` and `main()` — then call `main()` from an `if __name__ == "__main__"` block at the bottom. Put these imports at the top:

```python
import cv2
import argparse
import os
import time
```

---

### `parse_args()` — define and read command-line arguments

**What to achieve:** Return an object that gives the rest of the script access to four user-supplied options as `args.person`, `args.count`, `args.camera`, and `args.auto_interval`.

Use `argparse.ArgumentParser` — look up `ArgumentParser` and `add_argument` in the **Python 3 `argparse` documentation**. Create one parser object, then call `.add_argument()` four times:

- `--person` is a `str` that must always be provided (`required=True`). It becomes both the display name on screen and the subfolder name under `data/raw/`, acting as the class label the CNN will learn later.
- `--count` is an `int` with a default of `50`. The capture loop exits automatically once this many images are saved.
- `--camera` is an `int` with a default of `0`. OpenCV numbers webcam devices starting at zero; most laptops have only the built-in webcam at index 0.
- `--auto-interval` is a `float` with a default of `0.5`. It sets the minimum gap in seconds between frames in auto-capture mode. Note that argparse converts the hyphen in `--auto-interval` to an underscore in the result, so access it as `args.auto_interval`.

End the function by returning `parser.parse_args()`. argparse reads `sys.argv` automatically — you pass nothing to it.

---

### `main()` — setup before the loop

**What to achieve:** Prepare the output folder, determine the starting image index, open the camera, and load the face detector — all before entering the capture loop.

**1. Create the output folder.** Build the path `data/raw/<PersonName>/` using `os.path.join("data", "raw", args.person)`. Then create it with `os.makedirs` — look up `os.makedirs` in the **Python 3 `os` module documentation**. Pass `exist_ok=True` so that if the folder already exists from a previous run, the call succeeds silently rather than raising an error and crashing before a single image is captured.

**2. Find the starting image index.** Call `os.listdir` on the output folder and take the `len` of the returned list. Assign this number to `img_index`. Starting from the existing count instead of zero means a second run continues numbering from where the first stopped, so no previously saved files are overwritten. `os.listdir` returns filenames only (not full paths), and it counts every file in the folder — this is fine because the folder will only ever contain `.jpg` files from this script.

**3. Open the webcam.** Call `cv2.VideoCapture(args.camera)` — look up `VideoCapture` in the **OpenCV Python documentation**. The constructor attempts to open the camera device at the given index and always returns a capture object, even when the device was not found. You must check separately by calling `.isOpened()` on the returned object. If it returns `False`, raise a `RuntimeError` with a message that includes which camera index was tried, so the problem is immediately obvious instead of silently producing no output.

**4. Load the face detector.** Instantiate `cv2.CascadeClassifier` — look up `CascadeClassifier` in the **OpenCV documentation**. OpenCV ships with pre-trained Haar cascade XML files; you need the frontal-face one. Pass `cv2.data.haarcascades + "haarcascade_frontalface_default.xml"` as the path. `cv2.data.haarcascades` is a string pointing to the directory where OpenCV installed its bundled data files, so this path works on every machine regardless of where OpenCV is installed. This detector is used only for the live preview bounding boxes — it plays no role in saving images.

**5. Declare state variable.** Before the loop, set `last_capture_time = time.time()`. The script always runs in auto-capture mode, so no mode flag or per-frame trigger flag is needed.

---

### `main()` — the capture loop

**What to achieve:** Show a live annotated camera feed and save raw frames to disk at a fixed time interval. Use `while img_index < args.count` so the loop exits automatically once the target number of images is reached.

**6. Read a frame.** Call `cap.read()` at the top of every iteration — look up `VideoCapture.read` in the **OpenCV documentation**. It returns a tuple `(ret, frame)` where `ret` is `False` if the camera stopped sending frames. Break out of the loop immediately in that case to avoid processing empty data.

**7. Make a display copy.** Call `frame.copy()` to create a separate array called `display`. Draw all rectangles and text on `display`, never on `frame`. The reason: you save `frame` to disk later, and baked-in rectangles would appear in every training image, causing the CNN to learn from annotated data rather than raw faces.

**8. Detect faces for the preview.** Convert `frame` to grayscale with `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` — look up `cvtColor` in the **OpenCV documentation**. OpenCV reads video frames in BGR channel order, but the cascade detector expects a single-channel image. Then call `face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)` — look up `detectMultiScale` in the **OpenCV `CascadeClassifier` documentation** to understand what `scaleFactor` and `minNeighbors` control. The return value is a list of `(x, y, w, h)` tuples: `x` and `y` are the top-left corner pixel coordinates, `w` is the box width, and `h` is the height.

**9. Draw the face boxes.** Loop over the detected face list. For each `(x, y, w, h)`, call `cv2.rectangle` on `display` — look up `rectangle` in the **OpenCV drawing functions documentation**. Pass the top-left corner as `(x, y)`, the bottom-right corner as `(x + w, y + h)`, the BGR colour `(0, 255, 0)` for green, and a thickness of `2`. These boxes give the user visual feedback for framing their face; they are never saved.

**10. Draw the HUD.** Use `cv2.putText` to overlay status text on `display` — look up `putText` in the **OpenCV drawing functions documentation**. Show the person's name with a progress counter and the fixed label `"AUTO"` (e.g. `"Alice  12/50  AUTO"`). Position the text near the top-left corner, use `cv2.FONT_HERSHEY_SIMPLEX` as the font, and pick a scale and colour that are visible against a typical background.

**11. Show the frame.** Call `cv2.imshow("Capture", display)` to render the annotated frame, then `cv2.waitKey(1)` to flush the GUI event loop and keep the window responsive — look up `waitKey` in the **OpenCV documentation**.

**12. Implement auto-capture timing.** Compare `time.time()` against `last_capture_time` — look up `time.time` in the **Python 3 `time` module documentation**; it returns the current time as a float in seconds. If the elapsed time is greater than or equal to `args.auto_interval`, save the frame and update `last_capture_time = time.time()` so the next trigger is measured from this moment.

**13. Save the frame.** Build a filename in the format `<PersonName>_<NNNN>.jpg` where `NNNN` is `img_index` zero-padded to four digits. Look up **Python f-string format specifications** in the Python documentation — the format code `{value:04d}` means "integer, minimum width 4, pad with zeros on the left". Join the directory and filename with `os.path.join`, then write the image with `cv2.imwrite(filepath, frame)` — look up `imwrite` in the **OpenCV documentation**. Always write `frame`, not `display`. After saving, increment `img_index` by one.

**14. Release resources.** After the loop exits, call `cap.release()` to free the camera device and `cv2.destroyAllWindows()` to close the preview window — look up both in the **OpenCV documentation**. Always do this even when breaking early; skipping it leaves the camera locked and unavailable to other programs.

---

### Common mistakes

| Mistake | Why it matters |
|---|---|
| Drawing on `frame` instead of `display` | Saved images have rectangles baked in, which confuses the CNN during training |
| Not checking `cap.isOpened()` | The script silently does nothing when an invalid camera index is given |
| Starting `img_index` at `0` every run | Previous images get overwritten; always initialise from `len(os.listdir(out_dir))` |
| Writing `display` instead of `frame` to disk | The annotated copy gets saved instead of the clean original |

---

## Part A — File 2: `src/preprocessing/preprocess.py`

### How to run

```bash
python src/preprocessing/preprocess.py
```

The script automatically clears `data/processed/` and `data/splits/` before running, so it is safe to re-run at any time.

### File structure

Write three functions — `step1_crop_all()`, `step2_split()`, and `step3_augment_train()` — then call them in order from an `if __name__ == "__main__"` block. Before calling the three steps, clear both output directories so that stale files from a previous run cannot mix with the current output:

```python
if __name__ == "__main__":
    for directory in [PROCESSED_DIR, SPLITS_DIR]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
    step1_crop_all()
    step2_split()
    step3_augment_train()
``` Put these imports and constants at the top of the file before any function definition:

```python
import cv2
import os
import shutil
import random
from pathlib import Path
from PIL import Image, ImageEnhance

IMG_SIZE       = 128    # must match Part B
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
AUGMENT_FACTOR = 4
RANDOM_SEED    = 42

RAW_DIR        = "data/raw"
PROCESSED_DIR  = "data/processed"
SPLITS_DIR     = "data/splits"
```

---

### `step1_crop_all()` — detect, crop, resize, and save

**What to achieve:** For every raw image in `data/raw/<Person>/`, detect the face, crop a padded region around it, resize to `IMG_SIZE × IMG_SIZE` pixels, and save the result as a PNG into `data/processed/<Person>/`.

**Loading the cascade.** Instantiate `cv2.CascadeClassifier` using `cv2.data.haarcascades + "haarcascade_frontalface_default.xml"` as the file path — look up `CascadeClassifier` in the **OpenCV documentation**. `cv2.data.haarcascades` resolves to wherever OpenCV installed its bundled XML files, so this path works on every machine. Load the cascade once before the loops so the XML is not re-parsed on every image.

**Iterating over people and images.** Use `os.listdir(RAW_DIR)` to get person folder names — look up `os.listdir` in the **Python 3 `os` module documentation**. For each person, build the raw input path and the processed output path with `os.path.join`, then create the output folder with `os.makedirs(..., exist_ok=True)`. Inside that, call `os.listdir` again on the person's raw folder to iterate over individual image filenames.

**Reading each image.** Use `cv2.imread(filepath)` — look up `imread` in the **OpenCV documentation**. It returns a NumPy array on success and `None` if the file cannot be read. Check for `None` immediately and skip the file with `continue` if so. This prevents a crash from propagating through an entire person's folder just because of a stray `.DS_Store` or corrupt file.

**Detecting the face.** Convert the image to grayscale with `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`, then call `face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)` — look up `detectMultiScale` in the **OpenCV `CascadeClassifier` documentation**. The return value is a list of `(x, y, w, h)` bounding boxes.

**Choosing the crop region.** Handle two cases. If at least one face was detected (`len(faces) > 0`), take the first bounding box. A box that exactly wraps the face tends to clip the forehead or chin, so add 20% padding on all sides: compute the padding amount as `int(0.20 * max(w, h))`, then expand each edge of the box outward by that many pixels. Because the padded box may extend past the image boundary, clamp each coordinate using Python's built-in `max` and `min` — look up **Python built-in functions** in the Python documentation. The image width is `img.shape[1]` and the height is `img.shape[0]` (NumPy stores arrays as `(height, width, channels)`). After clamping, extract the crop with NumPy array slicing using `img[y1:y2, x1:x2]`. If no face was detected at all, use the entire image as the crop rather than discarding it — silently dropping images would shrink your dataset without any warning.

**Resizing.** Pass the crop to `cv2.resize(crop, (IMG_SIZE, IMG_SIZE))` — look up `resize` in the **OpenCV documentation**. The target size is given as `(width, height)`, so both values are `IMG_SIZE`. This normalises every image to the fixed spatial resolution the CNN will expect at training time.

**Saving as PNG.** Use `Path(fname).stem` from the `pathlib` module — look up `Path.stem` in the **Python 3 `pathlib` documentation** — to strip the original file extension, then append `.png`. Write the resized image with `cv2.imwrite(out_path, resized)`. PNG is lossless, so repeated reads and writes will not degrade image quality the way JPEG recompression would.

--- <!--here 0327 16:36-->

### `step2_split()` — divide images into train, val, and test sets

**What to achieve:** Randomly but reproducibly divide each person's processed images 70 / 15 / 15 across train, val, and test, then copy the files into the corresponding folders under `data/splits/`.

**Wiping the previous splits.** Delete the entire `data/splits/` directory with `shutil.rmtree` — look up `shutil.rmtree` in the **Python 3 `shutil` documentation** — then recreate the empty folder tree with `os.makedirs`. Do this once at the start of the function, before the per-person loop. Without this step, images from a previous run remain in the split folders and mix with the current ones, contaminating the dataset.

**Shuffling with a fixed seed.** After listing each person's `.png` files with `os.listdir`, call `random.seed(RANDOM_SEED)` and then `random.shuffle(files)` — look up both in the **Python 3 `random` module documentation**. Always place `random.seed` immediately before `random.shuffle`, not once at module load. Other code paths may call random functions in between runs and change the generator's internal state; re-seeding just before the shuffle guarantees every run on every machine produces exactly the same file ordering and therefore the same split.

**Computing split boundaries.** Calculate `n_train = int(len(files) * TRAIN_RATIO)` for the number of training images and `n_val = int(len(files) * VAL_RATIO)` for validation. Use Python list slicing to build three sub-lists — look up **list slicing** in the Python tutorial if the `list[start:end]` notation is unfamiliar. The training slice runs from the beginning to `n_train`; the validation slice runs from `n_train` to `n_train + n_val`; the test slice runs from `n_train + n_val` to the end with no upper bound needed.

**Copying files.** Loop over the three sub-lists alongside their corresponding split name (`"train"`, `"val"`, `"test"`). For each filename in a sub-list, build the source path from `PROCESSED_DIR` and the destination path from `SPLITS_DIR`, then copy with `shutil.copy(src, dst)` — look up `shutil.copy` in the **Python 3 `shutil` documentation**. Use `shutil.copy`, not `shutil.move`. Moving files empties `data/processed/`, making it impossible to regenerate the splits later without re-running step 1. The processed folder is the canonical store; the splits folder is derived from it and must be fully regenerable at any time.

---

### `step3_augment_train()` — generate augmented training copies

**What to achieve:** For every original image in `data/splits/train/<Person>/`, produce `AUGMENT_FACTOR` (4) additional variants using random geometric and colour transforms. This multiplies the effective training set size by five without capturing more photos.

**Filtering to originals only.** List the files in each person's training folder and keep only those whose name does not contain the substring `"_aug"`. This guard is essential: if the script is re-run without clearing the splits, augmented copies already exist in the folder and must not be augmented again. Augmenting augmented images would multiply the set by another factor and introduce near-duplicate images into training.

**Opening the image fresh on each copy.** For each original file, write a loop with `range(AUGMENT_FACTOR)`. Inside the loop, open the image at the start of every iteration with `Image.open(filepath).convert("RGB")` — look up `Image.open` and `Image.convert` in the **Pillow documentation**. Opening inside the inner loop is critical: if you opened once before the loop and applied transforms repeatedly to the same object, each copy would be a further-distorted version of the previous one rather than an independent variant of the original. The `.convert("RGB")` call is necessary because some PNG files contain an alpha (transparency) channel stored as RGBA; Pillow's enhancement classes require plain RGB input.

**Applying four transforms.** Apply the following in sequence on the same image object within each loop iteration:

- **Horizontal flip (50% chance):** Use `img.transpose(Image.FLIP_LEFT_RIGHT)` — look up `Image.transpose` in the **Pillow documentation**. Decide randomly with `random.random() < 0.5`. A horizontal flip simulates the same face being photographed from a slightly different angle, which is realistic.
- **Random rotation (−20° to +20°):** Use `img.rotate(degrees)` — look up `Image.rotate` in the **Pillow documentation**. Generate the angle with `random.uniform(-20, 20)` from the **Python `random` module**. This handles slight head tilts that a real webcam feed would show.
- **Random brightness (0.7× to 1.3×):** Use `ImageEnhance.Brightness(img).enhance(factor)` — look up `ImageEnhance` in the **Pillow documentation**. A factor of `1.0` leaves the image unchanged; values below `1.0` darken it and above `1.0` brighten it. Generate the factor with `random.uniform(0.7, 1.3)`. This simulates different lighting conditions across different recording sessions.
- **Random contrast (0.8× to 1.2×):** Use `ImageEnhance.Contrast(img).enhance(factor)` the same way, with `random.uniform(0.8, 1.2)`.

**Saving with a naming convention.** Build the output filename from the original file's stem plus a two-digit zero-padded copy index and the `_aug` marker — for example, `"Alice_0000_aug02.png"`. Use `Path(fname).stem` — look up `Path.stem` in the **Python 3 `pathlib` documentation** — to get the filename without its extension. Format the index with `f"{i:02d}"`. Save with `img.save(out_path)` — look up `Image.save` in the **Pillow documentation**. Saving into the same folder as the originals means `flow_from_directory` in Part B automatically picks up these copies during training.

---

### Expected output after a successful run

For a dataset with two people, each with 50 raw images:

```
data/
  processed/
    Alice/   ← 50 PNGs, 128×128
    Bob/     ← 50 PNGs, 128×128
  splits/
    train/
      Alice/ ← 35 originals + 140 _aug files = 175 total
      Bob/   ← 35 originals + 140 _aug files = 175 total
    val/
      Alice/ ← ~7 images
      Bob/   ← ~7 images
    test/
      Alice/ ← ~8 images
      Bob/   ← ~8 images
```

### Common mistakes

| Mistake | Why it matters |
|---|---|
| Not calling `random.seed` immediately before `random.shuffle` | Other code may change the random state between runs, producing a different split each time |
| Using `shutil.move` instead of `shutil.copy` | Files disappear from `data/processed/`; the splits cannot be regenerated without redoing step 1 |
| Swapping x and y when slicing the NumPy array | NumPy indexing is `[row, col]` = `[y, x]`; writing `img[x1:x2, y1:y2]` crops the wrong region |
| Not clamping the padded coordinates to image bounds | Negative indices wrap around in NumPy; out-of-range indices silently produce empty crops |
| Augmenting files whose names contain `"_aug"` | Re-running step 3 doubles the augmented set and introduces exact duplicates into training |
| Opening the image outside the per-copy inner loop | All copies receive the same transforms stacked on one object and become identical |
