import { spawn } from "child_process";
import fs from "fs/promises";
import path from "path";

import { DATA_ROOT, YOLO_CONF_THRESHOLD, YOLO_MODEL_PATH, YOLO_PYTHON, YOLO_SCRIPT_PATH } from "./constants";
import { readJob, writeJob } from "./jobStore";
import type { YoloImageResult, YoloJobResults } from "./types";

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
    proc.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(stderr || `yolo_detect.py exited with code ${code}`));
    });
  });
}

export async function runYoloOnJob(jobId: string): Promise<void> {
  const job = await readJob(jobId);
  if (!job) return;

  const yoloOutputDir = path.join(YOLO_DIR, jobId);

  // Mark as running
  await writeJob({
    ...job,
    yoloResults: {
      status: "running",
      model: path.basename(YOLO_MODEL_PATH),
      confThreshold: YOLO_CONF_THRESHOLD,
    },
  });

  try {
    await fs.mkdir(yoloOutputDir, { recursive: true });

    const originalImagePath = path.join(UPLOADS_DIR, jobId, "input.png");

    // Collect output image paths that actually exist locally
    const outputEntries: Array<{ key: string; filePath: string }> = [];
    for (const output of job.outputs) {
      if (output.source === "local") {
        const key = `${output.workflowName}_${output.variationIndex}`;
        outputEntries.push({ key, filePath: output.filePath });
      }
    }

    const allImagePaths = [
      originalImagePath,
      ...outputEntries.map((o) => o.filePath),
    ];

    await spawnYolo([
      "--model", YOLO_MODEL_PATH,
      "--images", ...allImagePaths,
      "--output-dir", yoloOutputDir,
      "--conf", String(YOLO_CONF_THRESHOLD),
    ]);

    const resultsRaw = await fs.readFile(
      path.join(yoloOutputDir, "results.json"),
      "utf8",
    );
    const results = JSON.parse(resultsRaw) as YoloDetectOutput;

    // Map original result
    const origBasename = "input.png";
    const origDet = results.detections[origBasename];
    const original: YoloImageResult = origDet
      ? {
          annotatedUrl: `/api/jobs/${jobId}/yolo/${encodeURIComponent(origDet.annotatedFile)}`,
          boxes: origDet.boxes,
        }
      : { annotatedUrl: "", boxes: [] };

    // Map output results
    const outputs: Record<string, YoloImageResult> = {};
    for (const entry of outputEntries) {
      const basename = path.basename(entry.filePath);
      const det = results.detections[basename];
      if (det) {
        outputs[entry.key] = {
          annotatedUrl: `/api/jobs/${jobId}/yolo/${encodeURIComponent(det.annotatedFile)}`,
          boxes: det.boxes,
        };
      }
    }

    const yoloResults: YoloJobResults = {
      status: "completed",
      model: results.model,
      confThreshold: results.confThreshold,
      original,
      outputs,
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
          error: (error as Error).message,
        },
      });
    }
  }
}
