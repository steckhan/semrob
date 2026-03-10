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

type YoloGtBox = {
  class: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
};

type IouMatch = {
  predIdx: number;
  gtIdx: number | null;
  iou: number;
};

type YoloImageResult = {
  annotatedUrl: string;
  boxes: YoloBox[];
  gtBoxes?: YoloGtBox[];
  iouMatches?: IouMatch[];
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

function iouColor(iou: number): string {
  if (iou >= 0.7) return "var(--green)";
  if (iou >= 0.5) return "var(--orange)";
  return "var(--red)";
}

function DetectionList({ result }: { result: YoloImageResult }) {
  const { boxes, iouMatches } = result;

  const matchMap = new Map<number, IouMatch>();
  if (iouMatches) {
    for (const m of iouMatches) {
      matchMap.set(m.predIdx, m);
    }
  }

  if (boxes.length === 0) return null;

  return (
    <div className="yolo-conf-list">
      {boxes.map((box, i) => {
        const match = matchMap.get(i);
        return (
          <div key={i} className="yolo-conf-row" style={{ flexWrap: "wrap", gap: "4px 8px" }}>
            <span className="yolo-conf-label">
              #{i + 1} {className(box.class)}
            </span>
            <span className="yolo-conf-value" style={{ color: confidenceColor(box.confidence) }}>
              conf: {(box.confidence * 100).toFixed(1)}%
            </span>
            {match !== undefined && (
              match.gtIdx !== null ? (
                <span style={{ fontSize: "0.7rem", color: iouColor(match.iou) }}>
                  IoU: {match.iou.toFixed(2)} ✓
                </span>
              ) : (
                <span style={{ fontSize: "0.7rem", color: "var(--red)" }}>
                  IoU: — FP
                </span>
              )
            )}
          </div>
        );
      })}
    </div>
  );
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

  const hasGt = !!(originalResult.gtBoxes?.length || inpaintedResult.gtBoxes?.length);

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

        {/* Legend */}
        {hasGt && (
          <div style={{ padding: "6px 16px", fontSize: "0.68rem", color: "var(--text-dim)", display: "flex", gap: 16 }}>
            <span><span style={{ color: "#6496ff", fontWeight: 700 }}>■</span> Ground truth (GT)</span>
            <span><span style={{ color: "var(--green)", fontWeight: 700 }}>■</span> Prediction conf ≥ 0.5</span>
            <span><span style={{ color: "var(--orange)", fontWeight: 700 }}>■</span> Prediction conf &lt; 0.5</span>
          </div>
        )}

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
                {originalResult.gtBoxes && originalResult.gtBoxes.length > 0 && (
                  <span style={{ marginLeft: 6, color: "#6496ff" }}>
                    {originalResult.gtBoxes.length} GT
                  </span>
                )}
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
            {originalResult.boxes.length > 0 && (
              <DetectionList result={originalResult} />
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
                {inpaintedResult.gtBoxes && inpaintedResult.gtBoxes.length > 0 && (
                  <span style={{ marginLeft: 6, color: "#6496ff" }}>
                    {inpaintedResult.gtBoxes.length} GT
                  </span>
                )}
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
            {inpaintedResult.boxes.length > 0 ? (
              <DetectionList result={inpaintedResult} />
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
