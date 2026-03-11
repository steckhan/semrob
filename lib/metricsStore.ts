import fs from "fs/promises";
import path from "path";

import { DATA_ROOT } from "./constants";
import type { AccumulatedMetrics, MetricsBucket } from "./types";

const METRICS_PATH = path.join(DATA_ROOT, "yolo", "metrics_accumulator.json");

type FrameEntry = {
  ap: number;
  precision: number;
  recall: number;
  f1: number;
  tp: number;
  fp: number;
  fn: number;
  iouSum: number;        // sum of IoU scores of TP matches in this frame
  tpMatchCount: number;  // number of TP matches in this frame (for IoU denominator)
  confSum: number;       // sum of confidence scores of all predicted boxes in this frame
  boxCount: number;      // total predicted boxes in this frame (for confidence denominator)
  timestamp: string;
};

type Bucket = {
  frames: Record<string, FrameEntry>;
  mAP: number;
  mAR: number;
  globalF1: number;
  totalTP: number;
  totalFP: number;
  totalFN: number;
  frameCount: number;
  // Micro-averaged across all frames (sum then divide once)
  totalIouSum: number;
  totalTpMatchCount: number;
  totalConfSum: number;
  totalBoxCount: number;
  meanIoU: number;        // computed: totalIouSum / totalTpMatchCount
  meanConfidence: number; // computed: totalConfSum / totalBoxCount
};

type MetricsAccumulator = {
  inpainted: Bucket;
  original: Bucket;
};

function emptyBucket(): Bucket {
  return { frames: {}, mAP: 0, mAR: 0, globalF1: 0, totalTP: 0, totalFP: 0, totalFN: 0, frameCount: 0, totalIouSum: 0, totalTpMatchCount: 0, totalConfSum: 0, totalBoxCount: 0, meanIoU: 0, meanConfidence: 0 };
}

async function readAccumulator(): Promise<MetricsAccumulator> {
  try {
    const raw = await fs.readFile(METRICS_PATH, "utf8");
    const parsed = JSON.parse(raw) as Partial<MetricsAccumulator>;
    return {
      inpainted: parsed.inpainted ?? emptyBucket(),
      original: parsed.original ?? emptyBucket(),
    };
  } catch {
    return { inpainted: emptyBucket(), original: emptyBucket() };
  }
}

async function writeAccumulator(data: MetricsAccumulator): Promise<void> {
  await fs.mkdir(path.dirname(METRICS_PATH), { recursive: true });
  await fs.writeFile(METRICS_PATH, JSON.stringify(data, null, 2));
}

function recompute(bucket: Bucket): void {
  const entries = Object.values(bucket.frames);
  const n = entries.length;
  bucket.mAP = n > 0 ? entries.reduce((s, e) => s + e.ap, 0) / n : 0;
  bucket.mAR = n > 0 ? entries.reduce((s, e) => s + e.recall, 0) / n : 0;
  // Use ?? 0 to handle legacy entries missing tp/fp/fn fields
  bucket.totalTP = entries.reduce((s, e) => s + (e.tp ?? 0), 0);
  bucket.totalFP = entries.reduce((s, e) => s + (e.fp ?? 0), 0);
  bucket.totalFN = entries.reduce((s, e) => s + (e.fn ?? 0), 0);
  const denom = 2 * bucket.totalTP + bucket.totalFP + bucket.totalFN;
  bucket.globalF1 = denom > 0 ? (2 * bucket.totalTP) / denom : 0;
  bucket.frameCount = n;
  // Micro-average: sum raw values across all frames, divide once
  bucket.totalIouSum       = entries.reduce((s, e) => s + (e.iouSum ?? 0), 0);
  bucket.totalTpMatchCount = entries.reduce((s, e) => s + (e.tpMatchCount ?? 0), 0);
  bucket.totalConfSum      = entries.reduce((s, e) => s + (e.confSum ?? 0), 0);
  bucket.totalBoxCount     = entries.reduce((s, e) => s + (e.boxCount ?? 0), 0);
  bucket.meanIoU        = bucket.totalTpMatchCount > 0 ? bucket.totalIouSum / bucket.totalTpMatchCount : 0;
  bucket.meanConfidence = bucket.totalBoxCount     > 0 ? bucket.totalConfSum / bucket.totalBoxCount    : 0;
}

/**
 * Record AP, Precision, Recall, F1 and raw counts for a frame in the given bucket.
 * frameId should be the original filename stem (e.g. "frame_000418").
 * bucket: "inpainted" (primary) or "original" (secondary baseline).
 */
export async function appendFrameMetrics(
  frameId: string,
  bucket: "inpainted" | "original",
  ap: number,
  precision: number,
  recall: number,
  f1: number,
  tp: number,
  fp: number,
  fn: number,
  iouSum = 0,
  tpMatchCount = 0,
  confSum = 0,
  boxCount = 0,
): Promise<void> {
  const data = await readAccumulator();
  data[bucket].frames[frameId] = { ap, precision, recall, f1, tp, fp, fn, iouSum, tpMatchCount, confSum, boxCount, timestamp: new Date().toISOString() };
  recompute(data[bucket]);
  await writeAccumulator(data);
}

/** Reset both buckets to empty — call at the start of each new batch run. */
export async function resetMetrics(): Promise<void> {
  await writeAccumulator({ inpainted: emptyBucket(), original: emptyBucket() });
}

/** Get accumulated metrics for both buckets across all sessions. */
export async function getAccumulatedMetrics(): Promise<AccumulatedMetrics> {
  const data = await readAccumulator();
  const toBucket = (b: Bucket): MetricsBucket => ({
    mAP: b.mAP,
    mAR: b.mAR,
    globalF1: b.globalF1,
    totalTP: b.totalTP,
    totalFP: b.totalFP,
    totalFN: b.totalFN,
    frameCount: b.frameCount,
    meanIoU: b.meanIoU ?? 0,
    meanConfidence: b.meanConfidence ?? 0,
  });
  return {
    inpainted: toBucket(data.inpainted),
    original: toBucket(data.original),
  };
}
