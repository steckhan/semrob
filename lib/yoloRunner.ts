import { spawn } from "child_process";
import fs from "fs/promises";
import path from "path";

import { DATA_ROOT, GT_DIR, YOLO_CONF_THRESHOLD, YOLO_MODEL_PATH, YOLO_PYTHON, YOLO_SCRIPT_PATH } from "./constants";
import { readJob, writeJob } from "./jobStore";
import { appendFrameMetrics } from "./metricsStore";
import { computeAP, computeCounts, computeF1, computePrecision, computeRecall, matchPredictionsToGT } from "./yoloMetrics";
import type { YoloGtBox, YoloImageResult, YoloJobResults } from "./types";

const UPLOADS_DIR = path.join(DATA_ROOT, "uploads");
const YOLO_DIR = path.join(DATA_ROOT, "yolo");

type YoloDetectOutput = {
  model: string;
  confThreshold: number;
  detections: Record<
    string,
    {
      sourcePath: string;
      annotatedFile: string;
      boxes: Array<{
        class: number;
        confidence: number;
        cx: number;
        cy: number;
        w: number;
        h: number;
      }>;
      gtBoxes?: Array<{
        class: number;
        cx: number;
        cy: number;
        w: number;
        h: number;
      }>;
    }
  >;
};

function spawnYolo(args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const proc = spawn(YOLO_PYTHON, [YOLO_SCRIPT_PATH, ...args]);
    let stderr = "";
    proc.stderr.on("data", (d: Buffer) => {
      stderr += d.toString();
    });
    // Handle spawn errors (e.g. ENOENT when YOLO_PYTHON path is wrong).
    // Without this, an unhandled EventEmitter error crashes the process on Node.js 24.
    proc.on("error", (err) => {
      reject(new Error(`Failed to spawn YOLO process: ${err.message}. Check YOLO_PYTHON="${YOLO_PYTHON}" and YOLO_SCRIPT_PATH="${YOLO_SCRIPT_PATH}".`));
    });
    proc.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(stderr || `yolo_detect.py exited with code ${code}`));
    });
  });
}

/** Check whether a GT .txt file exists for the given image filename (case-insensitive). */
async function gtFileExists(imageFilename: string): Promise<boolean> {
  const basename = path.parse(imageFilename).name.toLowerCase();
  try {
    const entries = await fs.readdir(GT_DIR);
    return entries.some(
      (e) => e.toLowerCase() === `${basename}.txt`,
    );
  } catch {
    return false;
  }
}

export async function runYoloOnJob(jobId: string): Promise<void> {
  const job = await readJob(jobId);
  if (!job) return;

  const yoloOutputDir = path.join(YOLO_DIR, jobId);

  // Determine GT availability based on original filename
  const originalFilename = job.originalFilename ?? "";
  const gtAvailable = originalFilename ? await gtFileExists(originalFilename) : false;

  // Mark as running
  await writeJob({
    ...job,
    yoloResults: {
      status: "running",
      model: path.basename(YOLO_MODEL_PATH),
      confThreshold: YOLO_CONF_THRESHOLD,
      gtAvailable,
    },
  });

  try {
    await fs.mkdir(yoloOutputDir, { recursive: true });

    const originalImagePath = path.join(UPLOADS_DIR, jobId, "input.png");

    // Collect output image paths that actually exist locally
    const outputEntries: Array<{ key: string; filePath: string }> = [];
    for (const output of job.outputs) {
      if (output.source === "local" && !output.filename?.startsWith("mask_")) {
        const key = `${output.workflowName}_${output.variationIndex}`;
        outputEntries.push({ key, filePath: output.filePath });
      }
    }

    const allImagePaths = [
      originalImagePath,
      ...outputEntries.map((o) => o.filePath),
    ];

    const yoloArgs = [
      "--model", YOLO_MODEL_PATH,
      "--images", ...allImagePaths,
      "--output-dir", yoloOutputDir,
      "--conf", String(YOLO_CONF_THRESHOLD),
    ];

    if (gtAvailable) {
      yoloArgs.push("--gt-dir", GT_DIR);
      yoloArgs.push("--gt-name", originalFilename);
    }

    await spawnYolo(yoloArgs);

    const resultsRaw = await fs.readFile(
      path.join(yoloOutputDir, "results.json"),
      "utf8",
    );
    const results = JSON.parse(resultsRaw) as YoloDetectOutput;

    // Map original result
    const origBasename = "input.png";
    const origDet = results.detections[origBasename];
    const origGtBoxes: YoloGtBox[] = origDet?.gtBoxes ?? [];
    const origIouMatches = origDet
      ? matchPredictionsToGT(origDet.boxes, origGtBoxes)
      : [];

    const original: YoloImageResult = origDet
      ? {
          annotatedUrl: `/api/jobs/${jobId}/yolo/${encodeURIComponent(origDet.annotatedFile)}`,
          boxes: origDet.boxes,
          gtBoxes: origGtBoxes.length > 0 ? origGtBoxes : undefined,
          iouMatches: origGtBoxes.length > 0 ? origIouMatches : undefined,
        }
      : { annotatedUrl: "", boxes: [] };

    // Compute original image metrics vs GT (secondary baseline)
    let frameAP: number | undefined;
    let framePrecision: number | undefined;
    let frameRecall: number | undefined;
    let frameF1: number | undefined;
    let frameTp: number | undefined;
    let frameFp: number | undefined;
    let frameFn: number | undefined;
    let frameMeanIoU: number | undefined;
    let frameMeanConfidence: number | undefined;
    if (gtAvailable && origDet) {
      frameAP        = computeAP(origDet.boxes, origGtBoxes);
      framePrecision = computePrecision(origDet.boxes, origGtBoxes);
      frameRecall    = computeRecall(origDet.boxes, origGtBoxes);
      frameF1        = computeF1(origDet.boxes, origGtBoxes);
      const origCounts = computeCounts(origDet.boxes, origGtBoxes);
      frameTp = origCounts.tp;
      frameFp = origCounts.fp;
      frameFn = origCounts.fn;
      // Raw sums for micro-averaged batch IoU and confidence
      const origTpMatches = origIouMatches.filter(m => m.gtIdx !== null);
      const origIouSum    = origTpMatches.reduce((s, m) => s + m.iou, 0);
      const origConfSum   = origDet.boxes.reduce((s, b) => s + b.confidence, 0);
      const origBoxCount  = origDet.boxes.length;
      // Per-frame means (used in single mode display)
      frameMeanIoU        = origTpMatches.length > 0 ? origIouSum / origTpMatches.length : 0;
      frameMeanConfidence = origBoxCount > 0 ? origConfSum / origBoxCount : 0;
      const frameId = path.parse(originalFilename).name;
      await appendFrameMetrics(frameId, "original", frameAP, framePrecision, frameRecall, frameF1, origCounts.tp, origCounts.fp, origCounts.fn, origIouSum, origTpMatches.length, origConfSum, origBoxCount).catch(() => {});
    }

    // Map output results
    const outputs: Record<string, YoloImageResult> = {};
    for (const entry of outputEntries) {
      const basename = path.basename(entry.filePath);
      const det = results.detections[basename];
      if (det) {
        const gtBoxes: YoloGtBox[] = det.gtBoxes ?? [];
        const iouMatches = gtBoxes.length > 0 ? matchPredictionsToGT(det.boxes, gtBoxes) : undefined;
        outputs[entry.key] = {
          annotatedUrl: `/api/jobs/${jobId}/yolo/${encodeURIComponent(det.annotatedFile)}`,
          boxes: det.boxes,
          gtBoxes: gtBoxes.length > 0 ? gtBoxes : undefined,
          iouMatches,
        };
      }
    }

    // Compute inpainted metrics vs GT (primary research metric) — average across all variants
    let inpaintedFrameAP: number | undefined;
    let inpaintedFramePrecision: number | undefined;
    let inpaintedFrameRecall: number | undefined;
    let inpaintedFrameF1: number | undefined;
    let inpaintedFrameTp: number | undefined;
    let inpaintedFrameFp: number | undefined;
    let inpaintedFrameFn: number | undefined;
    let inpaintedFrameMeanIoU: number | undefined;
    let inpaintedFrameMeanConfidence: number | undefined;
    if (gtAvailable && origGtBoxes.length >= 0 && outputEntries.length > 0) {
      const variantAPs: number[] = [];
      const variantPrecisions: number[] = [];
      const variantRecalls: number[] = [];
      const variantF1s: number[] = [];
      const variantTPs: number[] = [];
      const variantFPs: number[] = [];
      const variantFNs: number[] = [];
      // Raw sums across all variants (for micro-averaging in batch store)
      let totalIouSum = 0;
      let totalTpMatchCount = 0;
      let totalConfSum = 0;
      let totalBoxCount = 0;

      for (const entry of outputEntries) {
        const basename = path.basename(entry.filePath);
        const det = results.detections[basename];
        if (det) {
          variantAPs.push(computeAP(det.boxes, origGtBoxes));
          variantPrecisions.push(computePrecision(det.boxes, origGtBoxes));
          variantRecalls.push(computeRecall(det.boxes, origGtBoxes));
          variantF1s.push(computeF1(det.boxes, origGtBoxes));
          const c = computeCounts(det.boxes, origGtBoxes);
          variantTPs.push(c.tp);
          variantFPs.push(c.fp);
          variantFNs.push(c.fn);
          // Accumulate raw IoU and confidence sums across variants
          const varIouMatches = matchPredictionsToGT(det.boxes, origGtBoxes);
          const varTpMatches  = varIouMatches.filter(m => m.gtIdx !== null);
          totalIouSum      += varTpMatches.reduce((s, m) => s + m.iou, 0);
          totalTpMatchCount += varTpMatches.length;
          totalConfSum     += det.boxes.reduce((s, b) => s + b.confidence, 0);
          totalBoxCount    += det.boxes.length;
        }
      }

      if (variantAPs.length > 0) {
        const mean = (arr: number[]) => arr.reduce((s, v) => s + v, 0) / arr.length;
        inpaintedFrameAP        = mean(variantAPs);
        inpaintedFramePrecision = mean(variantPrecisions);
        inpaintedFrameRecall    = mean(variantRecalls);
        inpaintedFrameF1        = mean(variantF1s);
        inpaintedFrameTp        = mean(variantTPs);
        inpaintedFrameFp        = mean(variantFPs);
        inpaintedFrameFn        = mean(variantFNs);
        // Per-frame means for single mode display (avg across variants)
        inpaintedFrameMeanIoU        = totalTpMatchCount > 0 ? totalIouSum / totalTpMatchCount : 0;
        inpaintedFrameMeanConfidence = totalBoxCount     > 0 ? totalConfSum / totalBoxCount    : 0;
        const frameId = path.parse(originalFilename).name;
        await appendFrameMetrics(frameId, "inpainted", inpaintedFrameAP, inpaintedFramePrecision, inpaintedFrameRecall, inpaintedFrameF1, inpaintedFrameTp, inpaintedFrameFp, inpaintedFrameFn, totalIouSum, totalTpMatchCount, totalConfSum, totalBoxCount).catch(() => {});
      }
    }

    const yoloResults: YoloJobResults = {
      status: "completed",
      model: results.model,
      confThreshold: results.confThreshold,
      original,
      outputs,
      gtAvailable,
      frameAP,
      framePrecision,
      frameRecall,
      frameF1,
      frameTp,
      frameFp,
      frameFn,
      frameMeanIoU,
      frameMeanConfidence,
      inpaintedFrameAP,
      inpaintedFramePrecision,
      inpaintedFrameRecall,
      inpaintedFrameF1,
      inpaintedFrameTp,
      inpaintedFrameFp,
      inpaintedFrameFn,
      inpaintedFrameMeanIoU,
      inpaintedFrameMeanConfidence,
    };

    const updatedJob = await readJob(jobId);
    if (updatedJob) {
      await writeJob({ ...updatedJob, yoloResults });
    }
  } catch (error) {
    const updatedJob = await readJob(jobId);
    if (updatedJob) {
      await writeJob({
        ...updatedJob,
        yoloResults: {
          status: "failed",
          model: path.basename(YOLO_MODEL_PATH),
          confThreshold: YOLO_CONF_THRESHOLD,
          gtAvailable,
          error: (error as Error).message,
        },
      });
    }
  }
}
