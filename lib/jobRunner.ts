import fs from "fs/promises";
import http from "node:http";
import https from "node:https";
import path from "path";
import { randomUUID } from "crypto";

import sharp from "sharp";

import {
  COMFYUI_BASE_URL,
  DATA_ROOT,
  MAX_PARALLEL_WORKFLOWS,
} from "./constants";
import type {
  InpaintParams,
  JobOutput,
  JobRecord,
  WorkflowDefinition,
  WorkflowMapping,
} from "./types";
import { runWithConcurrency } from "./concurrency";
import { buildPatchedWorkflow, submitPatchedWorkflow, uploadImageToComfyUI, waitForWorkflow } from "./comfyuiClient";
import { ensureJobStore, updateJobStatus, writeJob } from "./jobStore";
import { runOpenAIInpainting } from "./openaiInpaintClient";
import { DEFAULT_PATCH_TARGETS } from "./workflowPatcher";
import { runYoloOnJob } from "./yoloRunner";

/**
 * Composite a grayscale mask (white=inpaint, black=keep) as the alpha channel
 * of the image. The resulting RGBA PNG has:
 *   - RGB: original image pixels
 *   - Alpha: inverted mask (white mask area → alpha=0 transparent, black → alpha=255 opaque)
 *
 * ComfyUI's LoadImage node extracts alpha=0 regions as the inpaint mask.
 */
async function compositeAlphaMask(
  imageBuffer: Buffer,
  maskBuffer: Buffer,
): Promise<Buffer> {
  // Get image dimensions
  const meta = await sharp(imageBuffer).metadata();
  const width = meta.width ?? 512;
  const height = meta.height ?? 512;

  // Extract mask as grayscale, flatten to remove any alpha, resize to match image.
  // The canvas exports a black-background PNG with white brush strokes.
  // Flatten ensures transparent canvas pixels become black (0 = keep area).
  const maskGray = await sharp(maskBuffer)
    .flatten({ background: { r: 0, g: 0, b: 0 } })
    .resize(width, height, { fit: "fill" })
    .grayscale()
    .raw()
    .toBuffer();

  // Flatten image to opaque RGB — removes any existing alpha channel so it
  // cannot accidentally contribute transparent pixels to the composite
  const imageRgb = await sharp(imageBuffer)
    .flatten({ background: { r: 255, g: 255, b: 255 } })
    .removeAlpha()
    .raw()
    .toBuffer();

  // Build RGBA buffer: image RGB + inverted mask as alpha
  // mask white (255) = inpaint area → alpha=0 (transparent) so LoadImage sees it as the mask
  // mask black (0) = keep area → alpha=255 (opaque)
  const rgba = Buffer.alloc(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    rgba[i * 4 + 0] = imageRgb[i * 3 + 0]; // R (3 channels after removeAlpha)
    rgba[i * 4 + 1] = imageRgb[i * 3 + 1]; // G
    rgba[i * 4 + 2] = imageRgb[i * 3 + 2]; // B
    rgba[i * 4 + 3] = 255 - maskGray[i];    // A: invert mask
  }

  return sharp(rgba, { raw: { width, height, channels: 4 } })
    .png()
    .toBuffer();
}

/**
 * Returns true when the workflow mapping uses the same node+key for image and
 * mask — i.e. the mask must be embedded as the alpha channel of the image.
 */
function isAlphaMaskWorkflow(mapping: WorkflowMapping | undefined): boolean {
  if (!mapping) return false;
  const t = mapping.targets;
  const imageKey = t.imageInputKey ?? "image";
  const maskKey = t.maskInputKey ?? "image";
  return t.maskNodeId === t.imageNodeId && maskKey === imageKey;
}

const UPLOADS_DIR = path.join(DATA_ROOT, "uploads");
const OUTPUTS_DIR = path.join(DATA_ROOT, "outputs");

function buildComfyViewUrl(
  comfyBaseUrl: string,
  filename: string,
  subfolder: string,
  imageType: string,
): string {
  return `${comfyBaseUrl}/view?filename=${encodeURIComponent(filename)}&subfolder=${encodeURIComponent(subfolder)}&type=${encodeURIComponent(imageType)}`;
}

function httpGetBuffer(url: string): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const client = url.startsWith("https://") ? https : http;
    client.get(url, (res) => {
      if (res.statusCode !== 200) {
        reject(new Error(`Failed to fetch ComfyUI output: HTTP ${res.statusCode}`));
        res.resume();
        return;
      }
      const chunks: Buffer[] = [];
      res.on("data", (chunk: Buffer) => chunks.push(chunk));
      res.on("end", () => resolve(Buffer.concat(chunks)));
      res.on("error", reject);
    }).on("error", reject);
  });
}

async function copyOutputToLocal(
  comfyBaseUrl: string,
  filename: string,
  subfolder: string,
  imageType: string,
  localPath: string,
): Promise<void> {
  const url = buildComfyViewUrl(comfyBaseUrl, filename, subfolder, imageType);
  const buffer = await httpGetBuffer(url);
  await fs.writeFile(localPath, buffer);
}

export type CreateJobInput = {
  imageBuffer: Buffer;
  maskBuffer: Buffer;
  params: InpaintParams;
  workflows: WorkflowDefinition[];
  mappings: WorkflowMapping[];
  comfyBaseUrl?: string;
  inpaintMode?: "local" | "api";
  openaiApiKey?: string;
  openaiModel?: string;
  originalFilename?: string;
};

export async function createJob({
  imageBuffer,
  maskBuffer,
  params,
  workflows,
  mappings,
  comfyBaseUrl,
  inpaintMode = "local",
  openaiApiKey,
  openaiModel,
  originalFilename,
}: CreateJobInput): Promise<JobRecord> {
  await ensureJobStore();

  const jobId = randomUUID();
  const jobDir = path.join(UPLOADS_DIR, jobId);
  await fs.mkdir(jobDir, { recursive: true });

  const imagePath = path.join(jobDir, "input.png");
  const maskPath = path.join(jobDir, "mask.png");

  await fs.writeFile(imagePath, imageBuffer);
  await fs.writeFile(maskPath, maskBuffer);

  // Upload images to ComfyUI via its HTTP API — cross-platform (no filesystem access needed)
  let comfyImagePathWindows = "";
  let comfyMaskPathWindows = "";
  if (inpaintMode !== "api") {
    const effectiveComfyBaseUrl = comfyBaseUrl ?? COMFYUI_BASE_URL;

    // If any workflow needs alpha-mask mode, upload a composited image too
    const needsAlpha = mappings.some(isAlphaMaskWorkflow);
    if (needsAlpha) {
      const composited = await compositeAlphaMask(imageBuffer, maskBuffer);
      await uploadImageToComfyUI(composited, "input_alpha.png", jobId, effectiveComfyBaseUrl);
    }

    // ComfyUI LoadImage expects paths relative to its input directory;
    // the upload API returns "subfolder/filename" which is exactly that.
    comfyImagePathWindows = await uploadImageToComfyUI(imageBuffer, "input.png", jobId, effectiveComfyBaseUrl);
    comfyMaskPathWindows = await uploadImageToComfyUI(maskBuffer, "mask.png", jobId, effectiveComfyBaseUrl);
  }

  const job: JobRecord = {
    id: jobId,
    createdAt: new Date().toISOString(),
    status: "queued",
    inpaintMode,
    openaiModel: inpaintMode === "api" ? (openaiModel ?? "gpt-image-1") : undefined,
    comfyBaseUrl: inpaintMode !== "api" ? (comfyBaseUrl ?? COMFYUI_BASE_URL) : undefined,
    originalFilename,
    params,
    workflows: workflows.map((workflow) => workflow.name),
    promptIds: {},
    outputs: [],
  };

  await writeJob(job);

  void runJob(
    job,
    workflows,
    mappings,
    comfyImagePathWindows,
    comfyMaskPathWindows,
    inpaintMode,
    openaiApiKey,
    openaiModel,
  ).catch(async (error) => {
    await updateJobStatus(job, "failed", (error as Error).message);
  });

  return job;
}

async function runOpenAIJob(job: JobRecord, apiKey: string, model?: string): Promise<void> {
  // Capture the returned record so startedAt is preserved in the completed call
  const runningJob = await updateJobStatus(job, "running");

  const jobDir = path.join(UPLOADS_DIR, job.id);
  const [imageBuffer, maskBuffer] = await Promise.all([
    fs.readFile(path.join(jobDir, "input.png")),
    fs.readFile(path.join(jobDir, "mask.png")),
  ]);

  const outputDir = path.join(OUTPUTS_DIR, job.id, "openai");
  await fs.mkdir(outputDir, { recursive: true });

  const results = await runOpenAIInpainting({
    imageBuffer,
    maskBuffer,
    prompt: runningJob.params.positivePrompt,
    apiKey,
    n: runningJob.params.variationCount,
    model: model ?? runningJob.openaiModel ?? "gpt-image-1",
  });

  const outputs: JobOutput[] = results.map((buffer, i) => {
    const filename = `output_${i}.png`;
    const filePath = path.join(outputDir, filename);
    return {
      workflowName: "openai",
      variationIndex: i,
      filePath,
      url: `/api/jobs/${job.id}/files/${encodeURIComponent("openai")}/${encodeURIComponent(filename)}`,
      source: "local" as const,
    };
  });

  // Write output files
  await Promise.all(
    results.map((buffer, i) => fs.writeFile(outputs[i].filePath, buffer)),
  );

  // Spread runningJob (has startedAt) so the completed record retains it
  await updateJobStatus({ ...runningJob, outputs }, "completed");
  void runYoloOnJob(job.id);
}

async function runJob(
  job: JobRecord,
  workflows: WorkflowDefinition[],
  mappings: WorkflowMapping[],
  imagePath: string,
  maskPath: string,
  inpaintMode?: "local" | "api",
  openaiApiKey?: string,
  openaiModel?: string,
): Promise<void> {
  if (inpaintMode === "api") {
    await runOpenAIJob(job, openaiApiKey ?? "", openaiModel);
    return;
  }

  // Capture the returned record so startedAt is preserved in the completed call
  const runningJob = await updateJobStatus(job, "running");
  await fs.mkdir(OUTPUTS_DIR, { recursive: true });

  const comfyBaseUrl = runningJob.comfyBaseUrl ?? COMFYUI_BASE_URL;

  const outputs: JobOutput[] = [];
  const promptIds: Record<string, string> = {};

  const promptList = runningJob.params.promptList?.filter((p) => p.trim());
  const effectiveVariationCount =
    promptList && promptList.length > 0
      ? promptList.length
      : (runningJob.params.variationCount ?? 1);
  const baseSeed = runningJob.params.seed;

  function seedForVariation(variationIndex: number): number {
    switch (runningJob.params.seedMode ?? "increment") {
      case "random":    return Math.floor(Math.random() * 0xFFFFFFFF);
      case "fixed":     return baseSeed;
      case "increment": return baseSeed + variationIndex;
    }
  }

  // Build all (workflow, variationIndex) pairs
  const workflowVariations: Array<{ workflow: (typeof workflows)[number]; variationIndex: number }> = [];
  for (const workflow of workflows) {
    for (let v = 0; v < effectiveVariationCount; v++) {
      workflowVariations.push({ workflow, variationIndex: v });
    }
  }

  await runWithConcurrency(workflowVariations, MAX_PARALLEL_WORKFLOWS, async ({ workflow, variationIndex }) => {
    const mapping = mappings.find((entry) => entry.workflowName === workflow.name);
    const variationParams = {
      ...runningJob.params,
      seed: seedForVariation(variationIndex),
      // Override positivePrompt per variation when a prompt sweep is active
      ...(promptList && promptList.length > 0
        ? { positivePrompt: promptList[variationIndex] ?? runningJob.params.positivePrompt }
        : {}),
    };

    // For alpha-mask workflows the image already has the mask composited into
    // its alpha channel; use the pre-written input_alpha.png instead of input.png
    const effectiveImagePath = isAlphaMaskWorkflow(mapping)
      ? imagePath.replace(/\/input\.png$/, "/input_alpha.png")
      : imagePath;

    const patchedWorkflow = buildPatchedWorkflow(
      {
        workflowName: workflow.name,
        workflowJson: workflow.json,
        targets: mapping?.targets ?? DEFAULT_PATCH_TARGETS,
      },
      effectiveImagePath,
      maskPath,
      variationParams,
    );
    const { promptId } = await submitPatchedWorkflow(patchedWorkflow, comfyBaseUrl);
    promptIds[`${workflow.name}:${variationIndex}`] = promptId;
    await writeJob({ ...runningJob, promptIds });

    const pollOutputs = await waitForWorkflow(promptId, comfyBaseUrl);
    for (const output of pollOutputs) {
      const filename = output.filename ?? path.basename(output.filePath);
      const localDir = path.join(OUTPUTS_DIR, runningJob.id, workflow.name);
      const localPath = path.join(localDir, filename);
      await fs.mkdir(localDir, { recursive: true });
      await copyOutputToLocal(
        comfyBaseUrl,
        filename,
        output.subfolder ?? "",
        output.imageType ?? "output",
        localPath,
      );

      outputs.push({
        ...output,
        workflowName: workflow.name,
        variationIndex,
        filePath: localPath,
        source: "local",
        url: `/api/jobs/${runningJob.id}/files/${encodeURIComponent(
          workflow.name,
        )}/${encodeURIComponent(filename)}`,
      });
    }
  });

  // Spread runningJob (has startedAt) so the completed record retains it
  await updateJobStatus(
    {
      ...runningJob,
      outputs,
      promptIds,
    },
    "completed",
  );
  void runYoloOnJob(job.id);
}
