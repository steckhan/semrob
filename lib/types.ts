export type InpaintMode = "inpaint" | "outpaint";

export type InpaintParams = {
  seed: number;
  steps: number;
  cfgScale: number;
  sampler: string;
  variationCount: number;
  useWorkflowDefaults: boolean;
  positivePrompt: string;
  colorMatchStrength: number;
  inpaintMode: InpaintMode;
};

export type WorkflowDefinition = {
  name: string;
  filePath: string;
  json: Record<string, unknown>;
};

export type WorkflowPatchTargets = {
  imageNodeId: string;
  imageInputKey?: string;
  /** Omit when the mask is embedded in the image as an alpha channel (e.g. flux2_klein). */
  maskNodeId?: string;
  maskInputKey?: string;
  paramsNodeId: string;
  positivePromptNodeId?: string;
  negativePromptNodeId?: string;
  /** Node ID for seed-only nodes (e.g. "easy seed"). Falls back to paramsNodeId. */
  seedNodeId?: string;
  /** Node ID for steps/scheduler nodes (e.g. Flux2Scheduler). Falls back to paramsNodeId. */
  stepsNodeId?: string;
  /** Node ID for sampler-select nodes (e.g. KSamplerSelect). Falls back to paramsNodeId. */
  samplerNodeId?: string;
  /** Node ID for CFG guider nodes (e.g. CFGGuider). Falls back to paramsNodeId. */
  cfgNodeId?: string;
  /** Node ID for color match post-processing (e.g. ColorMatch). */
  colorMatchNodeId?: string;
  /** Node ID for inpaint/outpaint mode switch (e.g. ImpactInt). */
  modeSwitchNodeId?: string;
};

export type WorkflowMapping = {
  workflowName: string;
  targets: WorkflowPatchTargets;
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
  error?: string;
};
