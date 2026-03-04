"use client";

import { useEffect } from "react";

type YoloBox = {
  class: number;
  confidence: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
};

type YoloImageResult = {
  annotatedUrl: string;
  boxes: YoloBox[];
};

type Props = {
  originalResult: YoloImageResult;
  inpaintedResult: YoloImageResult;
  inpaintedLabel: string;
  onClose: () => void;
};

const CLASS_NAMES: Record<number, string> = {
  0: "Hand",
  1: "Glove",
};

function className(cls: number): string {
  return CLASS_NAMES[cls] ?? `Class ${cls}`;
}

function confidenceColor(conf: number): string {
  if (conf >= 0.8) return "var(--green)";
  if (conf >= 0.5) return "var(--orange)";
  return "var(--red)";
}

export default function YoloCompareModal({
  originalResult,
  inpaintedResult,
  inpaintedLabel,
  onClose,
}: Props) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="compare-overlay" onClick={onClose}>
      <div
        className="yolo-modal"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="compare-header">
          <span>YOLO Detection — Original vs Inpainted</span>
          <button className="btn btn-outline btn-sm" onClick={onClose}>
            ✕ Close
          </button>
        </div>

        {/* Side-by-side images */}
        <div className="yolo-modal-images">
          <div className="yolo-modal-side">
            <div className="yolo-modal-side-header">
              <span className="yolo-modal-side-title">Original</span>
              <span
                className="yolo-modal-det-badge"
                style={{ color: originalResult.boxes.length > 0 ? "var(--red)" : "var(--green)" }}
              >
                {originalResult.boxes.length} detection{originalResult.boxes.length !== 1 ? "s" : ""}
              </span>
            </div>
            {originalResult.annotatedUrl ? (
              <img
                src={originalResult.annotatedUrl}
                alt="Original YOLO"
                className="yolo-modal-img"
              />
            ) : (
              <div className="yolo-modal-no-img">No annotated image</div>
            )}
            {/* Confidence list */}
            {originalResult.boxes.length > 0 && (
              <div className="yolo-conf-list">
                {originalResult.boxes.map((box, i) => (
                  <div key={i} className="yolo-conf-row">
                    <span className="yolo-conf-label">
                      #{i + 1} {className(box.class)}
                    </span>
                    <span
                      className="yolo-conf-value"
                      style={{ color: confidenceColor(box.confidence) }}
                    >
                      {(box.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="yolo-modal-divider" />

          <div className="yolo-modal-side">
            <div className="yolo-modal-side-header">
              <span className="yolo-modal-side-title">{inpaintedLabel}</span>
              <span
                className="yolo-modal-det-badge"
                style={{ color: inpaintedResult.boxes.length > 0 ? "var(--red)" : "var(--green)" }}
              >
                {inpaintedResult.boxes.length} detection{inpaintedResult.boxes.length !== 1 ? "s" : ""}
              </span>
            </div>
            {inpaintedResult.annotatedUrl ? (
              <img
                src={inpaintedResult.annotatedUrl}
                alt="Inpainted YOLO"
                className="yolo-modal-img"
              />
            ) : (
              <div className="yolo-modal-no-img">No annotated image</div>
            )}
            {/* Confidence list */}
            {inpaintedResult.boxes.length > 0 ? (
              <div className="yolo-conf-list">
                {inpaintedResult.boxes.map((box, i) => (
                  <div key={i} className="yolo-conf-row">
                    <span className="yolo-conf-label">
                      #{i + 1} {className(box.class)}
                    </span>
                    <span
                      className="yolo-conf-value"
                      style={{ color: confidenceColor(box.confidence) }}
                    >
                      {(box.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="hint" style={{ padding: "8px 0", color: "var(--green)" }}>
                No hands detected — glove concealment successful ✓
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
