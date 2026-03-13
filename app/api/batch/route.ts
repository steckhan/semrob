import { randomUUID } from "crypto";
import fs from "fs/promises";
import path from "path";

import { NextResponse } from "next/server";

import { COMFYUI_BASE_URL, DATA_ROOT, OPENAI_API_KEY } from "@/lib/constants";
import { writeBatch } from "@/lib/batchRunner";
import type { BatchRecord, BatchSubJob, InpaintParams } from "@/lib/types";

const BATCH_DIR = path.join(DATA_ROOT, "batch");

const DEFAULT_PARAMS: InpaintParams = {
  seed: 42,
  seedMode: "random",
  steps: 4,
  cfgScale: 1,
  sampler: "euler_ancestral",
  scheduler: "normal",
  denoise: 1,
  maskStrength: 1,
  variationCount: 1,
  useWorkflowDefaults: false,
  positivePrompt: "",
  negativePrompt: "",
  automaskMode: "manual",
  sam2Prompt: "hand",
  sam2Threshold: 0.39,
  unetName: "flux-2-klein-4b.safetensors",
};

/** POST /api/batch — create a new batch record, returns batchId */
export async function POST(request: Request) {
  const formData = await request.formData();

  const inpaintMode =
    String(formData.get("inpaintMode") ?? "local") === "api" ? "api" : "local";

  const rawComfyBaseUrl = String(formData.get("comfyBaseUrl") ?? "").trim();
  const comfyBaseUrl = rawComfyBaseUrl || COMFYUI_BASE_URL;

  const rawOpenaiModel = String(formData.get("openaiModel") ?? "gpt-image-1").trim();
  const openaiModel = rawOpenaiModel as string;

  const rawSeedMode = String(formData.get("seedMode") ?? DEFAULT_PARAMS.seedMode);
  const seedMode = (["random", "increment", "fixed"].includes(rawSeedMode)
    ? rawSeedMode
    : DEFAULT_PARAMS.seedMode) as InpaintParams["seedMode"];

  const params: InpaintParams = {
    seed: Number(formData.get("seed") ?? DEFAULT_PARAMS.seed),
    seedMode,
    steps: Number(formData.get("steps") ?? DEFAULT_PARAMS.steps),
    cfgScale: Number(formData.get("cfgScale") ?? DEFAULT_PARAMS.cfgScale),
    sampler: String(formData.get("sampler") ?? DEFAULT_PARAMS.sampler),
    scheduler: String(formData.get("scheduler") ?? DEFAULT_PARAMS.scheduler),
    denoise: Number(formData.get("denoise") ?? DEFAULT_PARAMS.denoise),
    maskStrength: Number(formData.get("maskStrength") ?? DEFAULT_PARAMS.maskStrength),
    variationCount: 1, // batch always does 1 variation per image
    useWorkflowDefaults:
      String(formData.get("useWorkflowDefaults") ?? "false") === "true",
    positivePrompt: String(formData.get("positivePrompt") ?? ""),
    negativePrompt: String(formData.get("negativePrompt") ?? ""),
    automaskMode: String(formData.get("automaskMode") ?? "manual") === "auto" ? "auto" : "manual",
    sam2Prompt: String(formData.get("sam2Prompt") ?? DEFAULT_PARAMS.sam2Prompt),
    sam2Threshold: Number(formData.get("sam2Threshold") ?? DEFAULT_PARAMS.sam2Threshold),
    unetName: String(formData.get("unetName") ?? DEFAULT_PARAMS.unetName),
  };

  const batchId = randomUUID();
  const batchImagesDir = path.join(BATCH_DIR, batchId);
  await fs.mkdir(batchImagesDir, { recursive: true });

  const batch: BatchRecord = {
    id: batchId,
    createdAt: new Date().toISOString(),
    status: "pending",
    totalImages: 0,
    completedImages: 0,
    failedImages: 0,
    inpaintMode,
    comfyBaseUrl: inpaintMode === "local" ? comfyBaseUrl : undefined,
    openaiModel: inpaintMode === "api" ? openaiModel : undefined,
    params,
    subJobs: [],
    outputDir: path.join(DATA_ROOT, "outputs"),
  };

  await writeBatch(batch);
  return NextResponse.json({ batchId, status: "pending" });
}
