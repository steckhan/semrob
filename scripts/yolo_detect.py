"""
YOLO Hand Detection Script
Runs YOLOv10 inference on one or more images, outputs:
  - results.json with detections per image
  - Annotated PNG images with bounding boxes + confidence labels
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = {0: "hand", 1: "glove"}


def load_gt_boxes(gt_dir: str, img_path: str) -> list[dict]:
    """Load YOLO-format ground truth boxes for the given image, if available."""
    if not gt_dir:
        return []
    basename = os.path.splitext(os.path.basename(img_path))[0]
    # Case-insensitive search for matching .txt file
    try:
        candidates = os.listdir(gt_dir)
    except OSError:
        return []
    for fname in candidates:
        if fname.lower() == f"{basename.lower()}.txt":
            gt_path = os.path.join(gt_dir, fname)
            gt_boxes = []
            try:
                with open(gt_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            gt_boxes.append({"class": cls_id, "cx": round(cx, 6), "cy": round(cy, 6), "w": round(bw, 6), "h": round(bh, 6)})
            except OSError:
                pass
            return gt_boxes
    return []


def run_detection(model_path: str, image_paths: list[str], output_dir: str, conf: float, gt_dir: str = "", gt_name: str = "") -> dict:
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)

    detections: dict[str, dict] = {}

    for img_path in image_paths:
        if not os.path.isfile(img_path):
            print(f"Warning: {img_path} not found, skipping.", file=sys.stderr)
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read {img_path}, skipping.", file=sys.stderr)
            continue

        h, w = img.shape[:2]
        results = model.predict(source=img_path, imgsz=640, conf=conf, verbose=False)
        result = results[0]

        boxes_data = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            # xyxy pixel coords
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # normalized center + size
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            boxes_data.append({
                "class": cls_id,
                "confidence": round(confidence, 4),
                "cx": round(cx, 6),
                "cy": round(cy, 6),
                "w": round(bw, 6),
                "h": round(bh, 6),
                "xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })

        # Load ground truth boxes — use gt_name override if provided (all images same frame)
        if gt_name:
            gt_lookup_path = os.path.join(os.path.dirname(img_path), gt_name)
            gt_boxes = load_gt_boxes(gt_dir, gt_lookup_path)
        else:
            gt_boxes = load_gt_boxes(gt_dir, img_path)

        # Draw annotated image
        annotated = img.copy()

        # Draw GT boxes first (blue, dashed-style with label)
        for gt in gt_boxes:
            gx1 = int((gt["cx"] - gt["w"] / 2) * w)
            gy1 = int((gt["cy"] - gt["h"] / 2) * h)
            gx2 = int((gt["cx"] + gt["w"] / 2) * w)
            gy2 = int((gt["cy"] + gt["h"] / 2) * h)
            gt_color = (255, 100, 0)  # BGR: blue-ish
            cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), gt_color, 2)
            gt_label = f"GT {CLASS_NAMES.get(gt['class'], str(gt['class']))}"
            (tw, th), _ = cv2.getTextSize(gt_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(annotated, (gx1, gy2), (gx1 + tw + 6, gy2 + th + 8), gt_color, -1)
            cv2.putText(annotated, gt_label, (gx1 + 3, gy2 + th + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw prediction boxes on top
        for det in boxes_data:
            x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
            conf_val = det["confidence"]
            color = (0, 255, 0) if conf_val >= 0.5 else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"hand {conf_val:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)

        # Build output filename from a sanitized key
        basename = os.path.splitext(os.path.basename(img_path))[0]
        annotated_filename = f"{basename}_annotated.png"
        annotated_path = os.path.join(output_dir, annotated_filename)
        cv2.imwrite(annotated_path, annotated)

        detections[os.path.basename(img_path)] = {
            "sourcePath": img_path,
            "annotatedFile": annotated_filename,
            "boxes": boxes_data,
            "gtBoxes": gt_boxes,
        }

    summary = {
        "model": os.path.basename(model_path),
        "confThreshold": conf,
        "detections": detections,
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="YOLO hand detection on images")
    parser.add_argument("--model", required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--images", nargs="+", required=True, help="Image paths")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--gt-dir", default="", help="Directory with ground truth .txt files (YOLO format)")
    parser.add_argument("--gt-name", default="", help="Original filename stem to use for GT lookup for all images (e.g. frame_000418.PNG)")
    args = parser.parse_args()

    summary = run_detection(args.model, args.images, args.output_dir, args.conf, args.gt_dir, args.gt_name)
    total_boxes = sum(len(d["boxes"]) for d in summary["detections"].values())
    print(f"Processed {len(summary['detections'])} images, {total_boxes} detections total.")


if __name__ == "__main__":
    main()
