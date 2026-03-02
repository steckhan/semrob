export type InpaintParams = {
  seed: number;
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
  status: JobStatus;
  inpaintMode?: "local" | "api";
  comfyBaseUrl?: string;
  params: InpaintParams;
  workflows: string[];
  promptIds: Record<string, string>;
  outputs: JobOutput[];
  patchedWorkflows?: Record<string, unknown>;
  error?: string;
};
