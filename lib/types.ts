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

export type YoloImageResult = {
  annotatedUrl: string;
  boxes: YoloBox[];
};

export type YoloJobResults = {
  status: "running" | "completed" | "failed";
  model: string;
  confThreshold: number;
  original?: YoloImageResult;
  outputs?: Record<string, YoloImageResult>;
  error?: string;
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
  params: InpaintParams;
  workflows: string[];
  promptIds: Record<string, string>;
  outputs: JobOutput[];
  patchedWorkflows?: Record<string, unknown>;
  yoloResults?: YoloJobResults;
  error?: string;
};
