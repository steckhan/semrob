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


def run_detection(model_path: str, image_paths: list[str], output_dir: str, conf: float) -> dict:
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

        # Draw annotated image
        annotated = img.copy()
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
    args = parser.parse_args()

    summary = run_detection(args.model, args.images, args.output_dir, args.conf)
    total_boxes = sum(len(d["boxes"]) for d in summary["detections"].values())
    print(f"Processed {len(summary['detections'])} images, {total_boxes} detections total.")


if __name__ == "__main__":
    main()
