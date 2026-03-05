import fs from "fs/promises";
import path from "path";

import { DATA_ROOT, COMFYUI_BASE_URL } from "./constants";
import type { BatchRecord, BatchSubJob, InpaintParams } from "./types";
import { createJob } from "./jobRunner";
import { runWithConcurrency } from "./concurrency";
import { loadWorkflowBundle } from "./workflowLoader";
import { runYoloOnJob } from "./yoloRunner";

const BATCH_DIR = path.join(DATA_ROOT, "batch");
const OUTPUTS_DIR = path.join(DATA_ROOT, "outputs");

export function batchPath(batchId: string): string {
  return path.join(BATCH_DIR, `${batchId}.json`);
}

export async function readBatch(batchId: string): Promise<BatchRecord | null> {
  try {
    const payload = await fs.readFile(batchPath(batchId), "utf8");
    return JSON.parse(payload) as BatchRecord;
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw err;
  }
}

export async function writeBatch(batch: BatchRecord): Promise<void> {
  await fs.mkdir(BATCH_DIR, { recursive: true });
  await fs.writeFile(batchPath(batch.id), JSON.stringify(batch, null, 2), "utf8");
}

/** Update completed/failed counts and overall status, then persist. */
async function updateBatchProgress(
  batch: BatchRecord,
  subJob: BatchSubJob,
): Promise<BatchRecord> {
  const completed = batch.subJobs.filter((j) => j.status === "completed").length;
  const failed = batch.subJobs.filter((j) => j.status === "failed").length;
  const total = batch.subJobs.length;

  const allDone = completed + failed >= total;
  const updated: BatchRecord = {
    ...batch,
    completedImages: completed,
    failedImages: failed,
    status: allDone ? (failed === total ? "failed" : "completed") : "running",
    completedAt: allDone ? new Date().toISOString() : undefined,
  };
  await writeBatch(updated);
  return updated;
}

/**
 * Run all images in a batch. Called asynchronously after the batch is created.
 * Processes up to BATCH_CONCURRENCY images in parallel.
 */
export async function runBatch(batchId: string): Promise<void> {
  const BATCH_CONCURRENCY = 4;

  let batch = await readBatch(batchId);
  if (!batch) throw new Error(`Batch ${batchId} not found`);

  // Mark batch as running
  batch = {
    ...batch,
    status: "running",
    startedAt: new Date().toISOString(),
  };
  await writeBatch(batch);

  const batchImagesDir = path.join(BATCH_DIR, batchId);

  const { workflows, mappings } = batch.inpaintMode === "local"
    ? await loadWorkflowBundle()
    : { workflows: [], mappings: [] };

  await runWithConcurrency(batch.subJobs, BATCH_CONCURRENCY, async (subJob) => {
    // Re-read latest batch to avoid stale state
    const current = (await readBatch(batchId))!;
    const idx = current.subJobs.findIndex((j) => j.imageIndex === subJob.imageIndex);

    try {
      const imagePath = path.join(batchImagesDir, `image_${subJob.imageIndex}.png`);
      const maskPath = path.join(batchImagesDir, `mask_${subJob.imageIndex}.png`);

      const [imageBuffer, maskBuffer] = await Promise.all([
        fs.readFile(imagePath),
        fs.readFile(maskPath),
      ]);

      current.subJobs[idx].status = "running";
      await writeBatch(current);

      const job = await createJob({
        imageBuffer,
        maskBuffer,
        params: current.params,
        workflows,
        mappings,
        comfyBaseUrl: current.inpaintMode === "local"
          ? (current.comfyBaseUrl ?? COMFYUI_BASE_URL)
          : undefined,
        inpaintMode: current.inpaintMode,
        openaiModel: current.inpaintMode === "api" ? current.openaiModel : undefined,
      });

      // Wait for the job to complete by polling the job file
      let attempts = 0;
      const MAX_POLL = 600; // up to ~15 min per image
      while (attempts < MAX_POLL) {
        await new Promise((res) => setTimeout(res, 1500));
        const { readJob } = await import("./jobStore");
        const latestJob = await readJob(job.id);
        if (!latestJob) break;
        if (latestJob.status === "completed" || latestJob.status === "failed") {
          const fresh = (await readBatch(batchId))!;
          const fi = fresh.subJobs.findIndex((j) => j.imageIndex === subJob.imageIndex);
          fresh.subJobs[fi] = {
            ...fresh.subJobs[fi],
            jobId: job.id,
            status: latestJob.status === "completed" ? "completed" : "failed",
            error: latestJob.error,
          };
          await updateBatchProgress(fresh, fresh.subJobs[fi]);
          // Fire-and-forget YOLO on completed sub-jobs
          if (latestJob.status === "completed") {
            void runYoloOnJob(job.id).catch(() => {});
          }
          return;
        }
        attempts++;
      }

      // Timed out
      const fresh = (await readBatch(batchId))!;
      const fi = fresh.subJobs.findIndex((j) => j.imageIndex === subJob.imageIndex);
      fresh.subJobs[fi] = {
        ...fresh.subJobs[fi],
        jobId: job.id,
        status: "failed",
        error: "Timed out waiting for job completion",
      };
      await updateBatchProgress(fresh, fresh.subJobs[fi]);
    } catch (err) {
      const fresh = (await readBatch(batchId))!;
      const fi = fresh.subJobs.findIndex((j) => j.imageIndex === subJob.imageIndex);
      fresh.subJobs[fi] = {
        ...fresh.subJobs[fi],
        status: "failed",
        error: (err as Error).message,
      };
      await updateBatchProgress(fresh, fresh.subJobs[fi]);
    }
  });

  // Final status update
  const final = (await readBatch(batchId))!;
  const allDone =
    final.subJobs.every((j) => j.status === "completed" || j.status === "failed");
  if (allDone && final.status === "running") {
    await writeBatch({
      ...final,
      status: "completed",
      completedAt: new Date().toISOString(),
    });
  }
}
