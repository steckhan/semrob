export type SeedMode = "random" | "increment" | "fixed";

export type InpaintParams = {
  seed: number;
  seedMode: SeedMode;
  steps: number;
  cfgScale: number;
  sampler: string;
  scheduler: string;
  denoise: number;
  maskStrength: number;
  variationCount: number;
  useWorkflowDefaults: boolean;
  positivePrompt: string;
  negativePrompt: string;
  automaskMode?: "manual" | "auto";
  sam2Prompt?: string;
  sam2Threshold?: number;
  unetName?: string;
};

export type WorkflowDefinition = {
  name: string;
  filePath: string;
  json: Record<string, unknown>;
};

export type WorkflowPatchTargets = {
  imageNodeId: string;
  imageInputKey?: string;
  maskNodeId: string;
  maskInputKey?: string;
  paramsNodeId: string;
  positivePromptNodeId?: string;
  negativePromptNodeId?: string;
  // Per-field node overrides — fall back to paramsNodeId if not set
  seedNodeId?: string;
  seedInputKey?: string;
  stepsNodeId?: string;
  stepsInputKey?: string;
  cfgNodeId?: string;
  cfgInputKey?: string;
  samplerNameNodeId?: string;
  samplerNameInputKey?: string;
  schedulerNodeId?: string;
  schedulerInputKey?: string;
  denoiseNodeId?: string;
  denoiseInputKey?: string;
  automaskSwitchNodeId?: string;
  sam2PromptNodeId?: string;
  unetNameNodeId?: string;
  unetNameInputKey?: string;
  clipNameNodeId?: string;
  clipNameInputKey?: string;
};

export type WorkflowMapping = {
  workflowName: string;
  targets: WorkflowPatchTargets;
};

export type YoloBox = {
  class: number;
  confidence: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
};

export type YoloGtBox = {
  class: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
};

export type IouMatch = {
  predIdx: number;
  gtIdx: number | null; // null = false positive (no GT match)
  iou: number;
};

export type YoloImageResult = {
  annotatedUrl: string;
  boxes: YoloBox[];
  gtBoxes?: YoloGtBox[];
  iouMatches?: IouMatch[];
};

export type YoloJobResults = {
  status: "running" | "completed" | "failed";
  model: string;
  confThreshold: number;
  original?: YoloImageResult;
  outputs?: Record<string, YoloImageResult>;
  gtAvailable?: boolean;
  // Original image vs GT (secondary baseline)
  frameAP?: number;
  framePrecision?: number;
  frameRecall?: number;
  frameF1?: number;
  frameTp?: number;
  frameFp?: number;
  frameFn?: number;
  frameMeanIoU?: number;         // avg IoU of TP matches, original image
  frameMeanConfidence?: number;  // avg confidence of all predicted boxes, original image
  // Inpainted image(s) vs GT — averaged across variants (primary research metric)
  inpaintedFrameAP?: number;
  inpaintedFramePrecision?: number;
  inpaintedFrameRecall?: number;
  inpaintedFrameF1?: number;
  inpaintedFrameTp?: number;
  inpaintedFrameFp?: number;
  inpaintedFrameFn?: number;
  inpaintedFrameMeanIoU?: number;        // avg IoU of TP matches, inpainted image(s)
  inpaintedFrameMeanConfidence?: number; // avg confidence of all predicted boxes, inpainted
  error?: string;
};

export type MetricsBucket = {
  mAP: number;
  mAR: number;
  globalF1: number;        // micro-averaged: 2·ΣTP / (2·ΣTP + ΣFP + ΣFN)
  totalTP: number;
  totalFP: number;
  totalFN: number;
  frameCount: number;
  meanIoU: number;         // avg IoU of TP-matched pairs only
  meanConfidence: number;  // avg confidence of all predicted boxes (TP + FP)
};

export type AccumulatedMetrics = {
  inpainted: MetricsBucket; // primary: how well inpainted images evade detector
  original: MetricsBucket;  // secondary: baseline detection on original images
};

export type JobStatus = "queued" | "running" | "completed" | "failed";

export type JobOutput = {
  workflowName: string;
  variationIndex: number;
  filePath: string;
  url: string;
  source: "comfyui" | "local";
  subfolder?: string;
  filename?: string;
  imageType?: string;
};

export type BatchStatus = "pending" | "uploading" | "running" | "completed" | "failed";

export type BatchSubJob = {
  imageIndex: number;
  originalName: string;
  jobId?: string;
  status: "pending" | "running" | "completed" | "failed";
  error?: string;
};

export type BatchRecord = {
  id: string;
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  status: BatchStatus;
  totalImages: number;
  completedImages: number;
  failedImages: number;
  inpaintMode: "local" | "api";
  comfyBaseUrl?: string;
  openaiModel?: string;
  params: InpaintParams;
  subJobs: BatchSubJob[];
  outputDir: string;
  error?: string;
};

export type JobRecord = {
  id: string;
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  status: JobStatus;
  inpaintMode?: "local" | "api";
  openaiModel?: string;
  comfyBaseUrl?: string;
  originalFilename?: string;
  params: InpaintParams;
  workflows: string[];
  promptIds: Record<string, string>;
  outputs: JobOutput[];
  patchedWorkflows?: Record<string, unknown>;
  yoloResults?: YoloJobResults;
  error?: string;
};
