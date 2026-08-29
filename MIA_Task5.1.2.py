import cv2
import numpy as np
from pathlib import Path

# =========================================================
# Folders
# =========================================================
IMAGE_DIR = Path("balls")
LABEL_DIR = Path("labels")
VIS_DIR = Path("visualizations")

LABEL_DIR.mkdir(exist_ok=True)
VIS_DIR.mkdir(exist_ok=True)

# =========================================================
# Classes
# =========================================================
BLUE = 0
RED = 1

# =========================================================
# Strict HSV Ranges (تضييق نطاق الألوان لمنع التقاط الجردل/السلك)
# =========================================================
# تم رفع الـ Saturation والـ Value لضمان أن اللون ناصع وخاص بالكورة فقط
RED_RANGES = [
    (np.array([0, 150, 70]), np.array([8, 255, 255])),
    (np.array([170, 150, 70]), np.array([180, 255, 255]))
]

BLUE_RANGE = (
    np.array([100, 150, 70]), # رفع الـ S من 40 لـ 150 لتجاهل لون الجردل الفاتح
    np.array([130, 255, 255])
)

# =========================================================
# Preprocessing & Masks
# =========================================================
def get_masks(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red Mask
    red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in RED_RANGES:
        red_mask |= cv2.inRange(hsv, lower, upper)

    # Blue Mask
    blue_mask = cv2.inRange(hsv, BLUE_RANGE[0], BLUE_RANGE[1])

    return red_mask, blue_mask

def clean_mask(mask):
    # دمج أجزاء الكورة المقصوصة بسبب النقوش السوداء
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close) # يقفل الفجوات السوداء داخل الكورة
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)   # يشيل أي نويز أو نقاط صغيرة
    return mask

# =========================================================
# Robust Ball Detection Core
# =========================================================
def detect_candidates(mask, class_id):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        
        # تجاهل المساحات الصغيرة جداً (أقل من حجم كورة حقيقية)
        if area < 800:
            continue

        # 1. اختبار نسبة شغل الكنتور للدائرة المحيطة (Solidity/Enclosing Circle)
        (x_c, y_c), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * (radius ** 2)
        
        if circle_area == 0:
            continue

        # نسبة مساحة الشكل المحسوب لمساحة أضغر دائرة محيطة بيه
        extent = area / circle_area

        # 2. حساب نسبة العرض إلى الارتفاع للـ Bounding Box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        # الكورة لازم تكون نسبتها قريبة من 1 (بين 0.7 و 1.3) ونسبة امتلائها الدائري عالية
        if 0.7 <= aspect_ratio <= 1.3 and extent > 0.40:
            candidates.append({
                "class_id": class_id,
                "x": int(x_c - radius),
                "y": int(y_c - radius),
                "w": int(2 * radius),
                "h": int(2 * radius),
                "score": extent * area
            })

    return candidates

# =========================================================
# Non-Maximum Suppression (NMS) / Duplicate Removal
# =========================================================
def iou(a, b):
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih

    union = (a["w"] * a["h"]) + (b["w"] * b["h"]) - intersection
    return intersection / (union + 1e-6)

def remove_duplicates(detections):
    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    final = []

    for det in detections:
        if any(iou(det, selected) > 0.20 for selected in final):
            continue
        final.append(det)

    return final

# =========================================================
# Save Labels & Visualization
# =========================================================
def save_labels(image_path, detections, width, height):
    label_path = LABEL_DIR / f"{image_path.stem}.txt"
    with open(label_path, "w") as f:
        for d in detections:
            # التأكد من إبقاء الإحداثيات داخل حدود الصورة
            x_min = max(0, d["x"])
            y_min = max(0, d["y"])
            w = min(width - x_min, d["w"])
            h = min(height - y_min, d["h"])

            xc = (x_min + w / 2.0) / width
            yc = (y_min + h / 2.0) / height
            w_norm = w / float(width)
            h_norm = h / float(height)

            f.write(f"{d['class_id']} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}\n")

def draw_results(image, detections):
    result = image.copy()
    for d in detections:
        color = (255, 0, 0) if d["class_id"] == BLUE else (0, 0, 255)
        name = "BLUE BALL" if d["class_id"] == BLUE else "RED BALL"

        x, y, w, h = max(0, d["x"]), max(0, d["y"]), d["w"], d["h"]

        cv2.rectangle(result, (x, y), (x + w, y + h), color, 3)
        cv2.putText(result, name, (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return result

# =========================================================
# Pipeline Execution
# =========================================================
def process_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return

    height, width = image.shape[:2]

    red_mask, blue_mask = get_masks(image)
    red_mask = clean_mask(red_mask)
    blue_mask = clean_mask(blue_mask)

    red_candidates = detect_candidates(red_mask, RED)
    blue_candidates = detect_candidates(blue_mask, BLUE)

    detections = remove_duplicates(red_candidates + blue_candidates)

    save_labels(image_path, detections, width, height)

    vis = draw_results(image, detections)
    cv2.imwrite(str(VIS_DIR / image_path.name), vis)

    print(f"Processed: {image_path.name} | Detected: {len(detections)}")

def main():
    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in extensions:
        image_paths.extend(IMAGE_DIR.glob(ext))

    image_paths = sorted(image_paths)
    print(f"Found {len(image_paths)} images.")

    for path in image_paths:
        process_image(path)

    print("\nCompleted successfully.")

if __name__ == "__main__":
    main()
