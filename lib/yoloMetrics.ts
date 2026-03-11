import type { IouMatch, YoloBox, YoloGtBox } from "./types";

/** IoU between two normalized YOLO boxes (cx, cy, w, h) */
export function boxIoU(a: YoloBox | { cx: number; cy: number; w: number; h: number }, b: YoloBox | { cx: number; cy: number; w: number; h: number }): number {
  const ax1 = a.cx - a.w / 2, ay1 = a.cy - a.h / 2;
  const ax2 = a.cx + a.w / 2, ay2 = a.cy + a.h / 2;
  const bx1 = b.cx - b.w / 2, by1 = b.cy - b.h / 2;
  const bx2 = b.cx + b.w / 2, by2 = b.cy + b.h / 2;

  const ix1 = Math.max(ax1, bx1), iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);

  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const aArea = a.w * a.h;
  const bArea = b.w * b.h;
  const union = aArea + bArea - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * Confidence-sorted matching: match predictions to GT boxes (same class, IoU >= threshold).
 * Processes predictions in descending confidence order — consistent with PASCAL VOC/COCO/YOLO eval
 * and with computeAP/computePrecision/computeRecall/computeCounts/computeF1.
 * Returns one IouMatch per prediction (gtIdx=null means false positive).
 */
export function matchPredictionsToGT(
  preds: YoloBox[],
  gts: Array<{ class: number; cx: number; cy: number; w: number; h: number }>,
  iouThresh = 0.5,
): IouMatch[] {
  // Sort by confidence descending, preserving original indices
  const indexed = preds.map((p, i) => ({ pred: p, origIdx: i }));
  indexed.sort((a, b) => b.pred.confidence - a.pred.confidence);

  const usedGt = new Set<number>();
  const matches: IouMatch[] = [];

  for (const { pred, origIdx } of indexed) {
    let bestIou = 0;
    let bestGi = -1;
    for (let gi = 0; gi < gts.length; gi++) {
      if (usedGt.has(gi) || pred.class !== gts[gi].class) continue;
      const iou = boxIoU(pred, gts[gi]);
      if (iou > bestIou) { bestIou = iou; bestGi = gi; }
    }
    if (bestIou >= iouThresh && bestGi >= 0) {
      matches.push({ predIdx: origIdx, gtIdx: bestGi, iou: bestIou });
      usedGt.add(bestGi);
    } else {
      matches.push({ predIdx: origIdx, gtIdx: null, iou: 0 });
    }
  }

  return matches;
}

/**
 * Compute Average Precision for one image.
 * preds sorted by confidence, gts are ground truth boxes.
 * Returns AP in [0, 1].
 */
export function computeAP(
  preds: YoloBox[],
  gts: Array<{ class: number; cx: number; cy: number; w: number; h: number }>,
  iouThresh = 0.5,
): number {
  if (gts.length === 0) return preds.length === 0 ? 1 : 0;
  if (preds.length === 0) return 0;

  // Sort predictions by confidence descending
  const sorted = [...preds].sort((a, b) => b.confidence - a.confidence);
  const usedGt = new Set<number>();

  let tp = 0, fp = 0;
  const precisions: number[] = [];
  const recalls: number[] = [];

  for (const pred of sorted) {
    let bestIou = 0;
    let bestGi = -1;
    for (let gi = 0; gi < gts.length; gi++) {
      if (usedGt.has(gi) || pred.class !== gts[gi].class) continue;
      const iou = boxIoU(pred, gts[gi]);
      if (iou > bestIou) { bestIou = iou; bestGi = gi; }
    }

    if (bestIou >= iouThresh && bestGi >= 0) {
      tp++;
      usedGt.add(bestGi);
    } else {
      fp++;
    }

    precisions.push(tp / (tp + fp));
    recalls.push(tp / gts.length);
  }

  // Area under PR curve (trapezoidal)
  let ap = recalls[0] * precisions[0];
  for (let i = 1; i < recalls.length; i++) {
    ap += (recalls[i] - recalls[i - 1]) * precisions[i];
  }
  return ap;
}

/**
 * Compute Recall for one image at a given IoU threshold.
 * Recall = TP / (TP + FN) = matched GT boxes / total GT boxes.
 * Returns value in [0, 1].
 */
export function computeRecall(
  preds: YoloBox[],
  gts: YoloGtBox[],
  iouThresh = 0.5,
): number {
  if (gts.length === 0) return preds.length === 0 ? 1 : 0;
  if (preds.length === 0) return 0;

  const usedGt = new Set<number>();
  let tp = 0;

  // Sort predictions by confidence descending (greedy highest-conf match first)
  const sorted = [...preds].sort((a, b) => b.confidence - a.confidence);
  for (const pred of sorted) {
    let bestIou = 0;
    let bestGi = -1;
    for (let gi = 0; gi < gts.length; gi++) {
      if (usedGt.has(gi) || pred.class !== gts[gi].class) continue;
      const iou = boxIoU(pred, gts[gi]);
      if (iou > bestIou) { bestIou = iou; bestGi = gi; }
    }
    if (bestIou >= iouThresh && bestGi >= 0) {
      tp++;
      usedGt.add(bestGi);
    }
  }

  return tp / gts.length;
}

/**
 * Compute Precision for one image at a given IoU threshold.
 * Precision = TP / (TP + FP) = TP / total predictions.
 * Returns value in [0, 1].
 */
export function computePrecision(
  preds: YoloBox[],
  gts: YoloGtBox[],
  iouThresh = 0.5,
): number {
  if (preds.length === 0) return gts.length === 0 ? 1 : 0; // no preds + GT exists = 0 precision (TP=0, FP=0 → undefined, treat as 0)
  if (gts.length === 0) return 0; // all predictions are FP

  const usedGt = new Set<number>();
  let tp = 0;

  const sorted = [...preds].sort((a, b) => b.confidence - a.confidence);
  for (const pred of sorted) {
    let bestIou = 0;
    let bestGi = -1;
    for (let gi = 0; gi < gts.length; gi++) {
      if (usedGt.has(gi) || pred.class !== gts[gi].class) continue;
      const iou = boxIoU(pred, gts[gi]);
      if (iou > bestIou) { bestIou = iou; bestGi = gi; }
    }
    if (bestIou >= iouThresh && bestGi >= 0) {
      tp++;
      usedGt.add(bestGi);
    }
  }

  return tp / preds.length;
}

/**
 * Compute raw TP/FP/FN counts for one image at a given IoU threshold.
 * Used for global (micro-averaged) F1 accumulation across frames.
 */
export function computeCounts(
  preds: YoloBox[],
  gts: YoloGtBox[],
  iouThresh = 0.5,
): { tp: number; fp: number; fn: number } {
  if (preds.length === 0 && gts.length === 0) return { tp: 0, fp: 0, fn: 0 };
  if (preds.length === 0) return { tp: 0, fp: 0, fn: gts.length };
  if (gts.length === 0) return { tp: 0, fp: preds.length, fn: 0 };

  const usedGt = new Set<number>();
  let tp = 0;

  const sorted = [...preds].sort((a, b) => b.confidence - a.confidence);
  for (const pred of sorted) {
    let bestIou = 0;
    let bestGi = -1;
    for (let gi = 0; gi < gts.length; gi++) {
      if (usedGt.has(gi) || pred.class !== gts[gi].class) continue;
      const iou = boxIoU(pred, gts[gi]);
      if (iou > bestIou) { bestIou = iou; bestGi = gi; }
    }
    if (bestIou >= iouThresh && bestGi >= 0) {
      tp++;
      usedGt.add(bestGi);
    }
  }

  return { tp, fp: preds.length - tp, fn: gts.length - tp };
}

/**
 * Compute F1 score for one image at a given IoU threshold.
 * F1 = 2·TP / (2·TP + FP + FN).
 * Returns value in [0, 1].
 */
export function computeF1(
  preds: YoloBox[],
  gts: YoloGtBox[],
  iouThresh = 0.5,
): number {
  if (gts.length === 0 && preds.length === 0) return 1;
  if (gts.length === 0) return 0; // FP only
  if (preds.length === 0) return 0; // FN only

  const usedGt = new Set<number>();
  let tp = 0;

  const sorted = [...preds].sort((a, b) => b.confidence - a.confidence);
  for (const pred of sorted) {
    let bestIou = 0;
    let bestGi = -1;
    for (let gi = 0; gi < gts.length; gi++) {
      if (usedGt.has(gi) || pred.class !== gts[gi].class) continue;
      const iou = boxIoU(pred, gts[gi]);
      if (iou > bestIou) { bestIou = iou; bestGi = gi; }
    }
    if (bestIou >= iouThresh && bestGi >= 0) {
      tp++;
      usedGt.add(bestGi);
    }
  }

  const fp = preds.length - tp;
  const fn = gts.length - tp;
  const denom = 2 * tp + fp + fn;
  return denom > 0 ? (2 * tp) / denom : 1;
}
