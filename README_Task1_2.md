# Task 1.2 – Detecting Red and Blue Balls (Classical Computer Vision)

## Problem
Given 20 images containing red and blue balls under varying lighting,
shadows, backgrounds, and positions, detect every ball, classify its color,
and output its location/size as a YOLO-format label file — using only
classical OpenCV techniques (no deep learning).

## My Approach

I broke the problem down into a pipeline of independent stages, so each
stage could be tuned/debugged on its own: **color segmentation → mask
cleanup → shape-based candidate filtering → duplicate removal → label
export**.

### 1. Color segmentation in HSV
I convert each image to HSV instead of working in RGB/BGR, because HSV
separates *color* (Hue) from *brightness* (Value) — this makes the color
detection far more robust to lighting changes and shadows, since a red ball
in shadow is still "red" in Hue even though it's darker.

Red wraps around hue 0, so I use **two hue ranges** (`0–8` and `170–180`)
combined with OR. Blue uses a single range around hue `100–130`.

**Key decision**: I deliberately used *strict* Saturation/Value thresholds
(`S ≥ 150, V ≥ 70`) instead of the usual looser ranges. While testing on the
actual images I noticed the background contains light-colored objects (a
bucket, a wire) that share the same hue as the balls but are much less
saturated/vivid — raising the S/V floor filters those out at the masking
stage itself, before any shape logic even runs. This turned out to be the
single biggest lever for reducing false positives.

### 2. Morphological cleanup
Real ball masks are rarely perfect blobs — patterned/textured balls with
dark markings can get fragmented into several small mask pieces. I clean
the mask in two steps:
- **Close** (large 15×15 kernel) first, to bridge small gaps/holes inside
  a ball caused by dark patterns or specular highlights.
- **Open** (smaller 5×5 kernel) after, to remove leftover small noise specks
  that survived the color threshold.

### 3. Shape-based candidate filtering
Color alone isn't enough — a red patch of background could pass the mask.
For every contour found in the cleaned mask, I compute:
- **Aspect ratio** of its bounding box — a ball should be close to square
  (`0.7`–`1.3`), which immediately rejects elongated blobs.
- **Extent** — the contour's area relative to its *minimum enclosing
  circle's* area. A real ball fills its enclosing circle almost completely;
  an irregular blob doesn't. I require this above `0.40`.
- A minimum area threshold, to reject tiny noise specks that are too small
  to plausibly be a ball.

I deliberately use the **minimum enclosing circle** (rather than the raw
bounding box) to build the final `(x, y, w, h)` box, since it directly
enforces the assumption that the object is round — this makes the final box
more consistent than just using the contour's bounding rectangle.

### 4. Removing duplicate detections (NMS)
Because a single ball can sometimes produce more than one passing contour
(e.g. after the morphological split), I run a simple **IoU-based
Non-Maximum Suppression**: sort all candidates by a confidence "score"
(circularity × area), then greedily keep a detection only if it doesn't
overlap an already-kept one by more than `20%` IoU.

### 5. Exporting YOLO-format labels
For each kept detection I convert the pixel box to normalized YOLO format
(`class_id x_center y_center width height`), clip it to stay inside the
image bounds, and write one `.txt` file per image with the same filename
stem as required.

### 6. Visualization
I also draw the final boxes back onto each image (color-coded — red boxes
for red balls, blue for blue balls) and save them separately, so I can
visually sanity-check the detections image-by-image instead of only
trusting the numbers.

## Why classical CV instead of anything else
The task explicitly requires a classical, non-deep-learning solution, so
the whole design is built around **combining a cheap, fast signal (HSV
color) with a geometric sanity-check (shape/roundness)** rather than trying
to make the color threshold alone perfect. Neither signal is reliable on
its own — color alone catches non-ball objects of a similar hue, and shape
alone can't tell color — but together they cut false positives a lot.

## Challenges & How I Handled Them
- **Background objects sharing ball colors** (bucket, wire): solved by
  tightening the Saturation/Value thresholds rather than the Hue range,
  since these objects are washed-out/light versions of the same hue.
- **Textured/patterned balls fragmenting into multiple contours**: solved
  with a large closing kernel to re-merge the pieces before contour
  detection.
- **Duplicate/overlapping detections**: solved with IoU-based NMS.
- **Boxes exceeding image bounds**: clipped explicitly when writing labels,
  so no negative or out-of-range values ever get written.

## Tools & Libraries
- `opencv-python` (`cv2`) – color space conversion, morphology, contour
  detection, `minEnclosingCircle`, drawing
- `numpy` – mask arithmetic and array operations
- `pathlib` – file/folder handling

## Output
- `labels/` – one YOLO-format `.txt` file per input image
- `labeled_images/` – same images with the detected balls drawn on them
- `labels.zip`, `labeled_images.zip` – zipped for submission
