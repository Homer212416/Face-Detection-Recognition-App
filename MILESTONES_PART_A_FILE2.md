# Part A — File 2 Milestones (`src/preprocessing/my_preprocess.py`)
Target: done before **17:00, 2026-03-29** — 6 × 30-min blocks starting 14:00

Mark each checkbox as you finish it: `- [x]`

---

## Current state

| Function                | Status      | Notes                                      |
|-------------------------|-------------|--------------------------------------------|
| `step1_crop_all()`      | Done        | Detect, pad, clamp, crop, resize, save PNG |
| `step2_split()`         | Partial     | Wipe + shuffle done; copy logic missing    |
| `step3_augment_train()` | Not started | Entirely commented out                     |
| `__main__` block        | Partial     | step2 and step3 calls commented out        |

---

## Milestone 1 — 14:00 → 14:30 · Finish `step2_split()` — split boundaries

Still inside the per-person `for p` loop, after `random.shuffle(files)`:

- [x] Compute `n_train = int(len(files) * TRAIN_RATIO)` — number of training images using the 0.70 constant
- [x] Compute `n_val = int(len(files) * VAL_RATIO)` — number of validation images using the 0.15 constant
- [x] Slice `train_files = files[:n_train]`
- [x] Slice `val_files   = files[n_train : n_train + n_val]`
- [x] Slice `test_files  = files[n_train + n_val :]` — no upper bound needed; remainder goes to test
- [x] Loop over `zip(["train", "val", "test"], [train_files, val_files, test_files])`
- [x] Inside that loop, build destination dir: `os.path.join(SPLITS_DIR, split_name, p)`
- [x] Create it with `os.makedirs(dst_dir, exist_ok=True)`
- [x] For each filename in the sub-list, build `src = os.path.join(PROCESSED_DIR, p, fname)`
- [x] Copy with `shutil.copy(src, dst_dir)` — use `shutil.copy`, **not** `shutil.move` (moving would empty `data/processed/` and make it impossible to regenerate splits later)
- [x] Uncomment `step2_split()` in the `__main__` block

---

## Milestone 2 — 14:30 → 15:00 · Smoke-test `step2_split()` + define `step3` skeleton

- [x] Run `python src/preprocessing/my_preprocess.py` and confirm no exceptions
- [ ] Confirm `data/splits/train/<Person>/`, `data/splits/val/<Person>/`, `data/splits/test/<Person>/` all exist and contain `.png` files
- [ ] Confirm file counts roughly match: ~35 train, ~7 val, ~8 test per person
- [ ] Confirm `data/processed/<Person>/` still has all its files (not emptied by a move)
- [ ] Uncomment / define `def step3_augment_train():` (remove the comment block)
- [ ] Call `random.seed(RANDOM_SEED)` **once before** the `for p` loop — not inside it, so the generator state advances naturally across people giving each person a different shuffle order
- [ ] Iterate over each person: `for p in os.listdir(os.path.join(SPLITS_DIR, "train"))`
- [ ] Build `train_person_dir = os.path.join(SPLITS_DIR, "train", p)`

---

## Milestone 3 — 15:00 → 15:30 · Filter originals + open image + write inner loop

- [ ] List all files in the person's train folder: `os.listdir(train_person_dir)`
- [ ] Filter to originals only — keep files whose name does **not** contain `"_aug"`: prevents augmenting already-augmented images on a re-run, which would multiply the set by another factor and introduce near-duplicates
  ```python
  originals = [f for f in files if "_aug" not in f]
  ```
- [ ] For each original filename, build `filepath = os.path.join(train_person_dir, fname)`
- [ ] Write the inner loop: `for i in range(AUGMENT_FACTOR):`
- [ ] Open the image fresh at the top of **every** inner-loop iteration:
  ```python
  img = Image.open(filepath).convert("RGB")
  ```
  Opening inside the loop is critical — opening once outside would stack all transforms onto one object, making every copy an increasingly distorted version of the previous one instead of an independent variant of the original. `.convert("RGB")` strips any alpha channel (RGBA) that PNG files may carry; Pillow's enhancement classes require plain RGB

---

## Milestone 4 — 15:30 → 16:00 · Apply four transforms

All four transforms are applied in sequence to `img` within each inner-loop iteration:

- [ ] **Horizontal flip (50% chance):** `if random.random() < 0.5: img = img.transpose(Image.FLIP_LEFT_RIGHT)` — simulates the same face photographed from a slightly different angle

- [ ] **Random rotation −20° to +20°:** `img = img.rotate(random.uniform(-20, 20))` — handles slight head tilts a real webcam feed would show

- [ ] **Random brightness 0.7× to 1.3×:** `img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))` — factor 1.0 = unchanged; below darkens, above brightens; simulates different lighting conditions

- [ ] **Random contrast 0.8× to 1.2×:** `img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))`

---

## Milestone 5 — 16:00 → 16:30 · Save augmented files + wire up `__main__`

- [ ] Build the output filename using `Path(fname).stem` to strip the extension, then append the copy index and `_aug` marker:
  ```python
  stem     = Path(fname).stem            # e.g. "Alice_0000"
  out_name = f"{stem}_aug{i:02d}.png"    # e.g. "Alice_0000_aug02.png"
  out_path = os.path.join(train_person_dir, out_name)
  ```
  `{i:02d}` means integer, minimum width 2, zero-padded on the left
- [ ] Save with `img.save(out_path)` — saving into the same folder as the originals means `flow_from_directory` in Part B picks them up automatically during training
- [ ] Uncomment `step2_split()` and `step3_augment_train()` in the `__main__` block
- [ ] Run the full pipeline: `python src/preprocessing/my_preprocess.py`

---

## Milestone 6 — 16:30 → 17:00 · Verify counts + re-run safety check

- [ ] Confirm `data/processed/<Person>/` — 50 PNGs, each 128×128
- [ ] Confirm `data/splits/train/<Person>/` — 35 originals + 140 `_aug` files = **175 total**
- [ ] Confirm `data/splits/val/<Person>/` — ~7 files, no `_aug` files
- [ ] Confirm `data/splits/test/<Person>/` — ~8 files, no `_aug` files
- [ ] Confirm `data/processed/<Person>/` is still intact (not emptied by a move)
- [ ] Re-run the script a second time and confirm totals stay the same — verifies the `__main__` wipe and the `"_aug"` filter both hold

---

## Traps to avoid

| Trap | Why it matters |
|---|---|
| `random.seed` called inside the per-person loop | Every person gets the same shuffle pattern; seed once before the loop so the generator state advances naturally and each person gets a different ordering |
| `shutil.move` instead of `shutil.copy` | Moves empty `data/processed/`; splits cannot be regenerated without redoing step 1 |
| Opening PIL image outside the `for i in range(AUGMENT_FACTOR)` loop | Each copy stacks transforms on the last — all copies end up as the same heavily-distorted image |
| Not filtering `"_aug"` files before augmenting | Re-running step 3 doubles the augmented set each time and fills training with near-duplicates |
| `img[x1:x2, y1:y2]` (x/y axes swapped) | NumPy indexing is `[row, col]` = `[y, x]`; swapping silently crops the wrong region |
