import fs from "fs/promises";
import path from "path";
import { randomUUID } from "crypto";

import {
  COMFYUI_BASE_URL,
  COMFYUI_INPUT_DIR,
  COMFYUI_INPUT_DIR_WINDOWS,
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
import { buildPatchedWorkflow, pollWorkflow, submitPatchedWorkflow } from "./comfyuiClient";
import { ensureJobStore, updateJobStatus, writeJob } from "./jobStore";
import { runOpenAIInpainting } from "./openaiInpaintClient";
import { DEFAULT_PATCH_TARGETS } from "./workflowPatcher";

const UPLOADS_DIR = path.join(DATA_ROOT, "uploads");
const OUTPUTS_DIR = path.join(DATA_ROOT, "outputs");

async function findComfyOutputPath(
  comfyBaseUrl: string,
  promptId: string,
  filename: string,
  subfolder: string | undefined,
  imageType: string | undefined,
): Promise<string> {
  if (subfolder && subfolder.length > 0) {
    return `${comfyBaseUrl}/view?filename=${encodeURIComponent(
      filename,
    )}&subfolder=${encodeURIComponent(subfolder)}&type=${encodeURIComponent(
      imageType ?? "output",
    )}`;
  }

  const response = await fetch(`${comfyBaseUrl}/history/${promptId}`);
  if (!response.ok) {
    throw new Error("Failed to locate ComfyUI output.");
  }

  const payload = (await response.json()) as Record<string, any>;
  const record = payload[promptId];
  if (!record?.outputs) {
    throw new Error("ComfyUI output not available yet.");
  }

  const match = Object.values(record.outputs)
    .flatMap((output: any) => output.images ?? [])
    .find((image: any) => image.filename === filename);

  if (!match) {
    throw new Error("ComfyUI output not available yet.");
  }

  const subfolderValue = match.subfolder ?? "";

  return `${comfyBaseUrl}/view?filename=${encodeURIComponent(
    filename,
  )}&subfolder=${encodeURIComponent(subfolderValue)}&type=${encodeURIComponent(
    imageType ?? "output",
  )}`;
}

async function copyOutputToLocal(
  comfyBaseUrl: string,
  promptId: string,
  filename: string,
  subfolder: string | undefined,
  imageType: string | undefined,
  localPath: string,
  attempts = 8,
  delayMs = 750,
): Promise<void> {
  let lastError: unknown;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const url = await findComfyOutputPath(
        comfyBaseUrl,
        promptId,
        filename,
        subfolder,
        imageType,
      );
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error("Failed to fetch ComfyUI output.");
      }

      const buffer = Buffer.from(await response.arrayBuffer());
      await fs.writeFile(localPath, buffer);
      return;
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }
  throw lastError;
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
}: CreateJobInput): Promise<JobRecord> {
  await ensureJobStore();

  const jobId = randomUUID();
  const jobDir = path.join(UPLOADS_DIR, jobId);
  await fs.mkdir(jobDir, { recursive: true });

  const imagePath = path.join(jobDir, "input.png");
  const maskPath = path.join(jobDir, "mask.png");

  await fs.writeFile(imagePath, imageBuffer);
  await fs.writeFile(maskPath, maskBuffer);

  // Only copy to ComfyUI input directory when running locally
  let comfyImagePathWindows = "";
  let comfyMaskPathWindows = "";
  if (inpaintMode !== "api") {
    const comfyJobDir = path.join(COMFYUI_INPUT_DIR, jobId);
    const comfyImagePath = path.join(comfyJobDir, "input.png");
    const comfyMaskPath = path.join(comfyJobDir, "mask.png");

    await fs.mkdir(comfyJobDir, { recursive: true });
    await fs.writeFile(comfyImagePath, imageBuffer);
    await fs.writeFile(comfyMaskPath, maskBuffer);

    comfyImagePathWindows = path
      .join(COMFYUI_INPUT_DIR_WINDOWS, jobId, "input.png")
      .replace(/\\/g, "/");
    comfyMaskPathWindows = path
      .join(COMFYUI_INPUT_DIR_WINDOWS, jobId, "mask.png")
      .replace(/\\/g, "/");
  }

  const job: JobRecord = {
    id: jobId,
    createdAt: new Date().toISOString(),
    status: "queued",
    inpaintMode,
    openaiModel: inpaintMode === "api" ? (openaiModel ?? "gpt-image-1") : undefined,
    comfyBaseUrl: inpaintMode !== "api" ? (comfyBaseUrl ?? COMFYUI_BASE_URL) : undefined,
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

  await runWithConcurrency(workflows, MAX_PARALLEL_WORKFLOWS, async (workflow) => {
    const mapping = mappings.find((entry) => entry.workflowName === workflow.name);
    const patchedWorkflow = buildPatchedWorkflow(
      {
        workflowName: workflow.name,
        workflowJson: workflow.json,
        targets: mapping?.targets ?? DEFAULT_PATCH_TARGETS,
      },
      imagePath,
      maskPath,
      runningJob.params,
    );
    const { promptId } = await submitPatchedWorkflow(patchedWorkflow, comfyBaseUrl);
    promptIds[workflow.name] = promptId;
    const patchedWorkflows = {
      ...(runningJob.patchedWorkflows ?? {}),
      [workflow.name]: patchedWorkflow,
    };
    await writeJob({ ...runningJob, promptIds, patchedWorkflows });

    let completed = false;
    while (!completed) {
      await new Promise((resolve) => setTimeout(resolve, 1500));
      const pollOutputs = await pollWorkflow(promptId, comfyBaseUrl);
      if (pollOutputs.length > 0) {
        for (const output of pollOutputs) {
          const filename = output.filename ?? path.basename(output.filePath);
          const subfolder = output.subfolder ?? "";
          const localDir = path.join(OUTPUTS_DIR, runningJob.id, workflow.name);
          const localPath = path.join(localDir, filename);
          await fs.mkdir(localDir, { recursive: true });
          await copyOutputToLocal(
            comfyBaseUrl,
            promptId,
            filename,
            output.subfolder,
            output.imageType,
            localPath,
          );

          outputs.push({
            ...output,
            workflowName: workflow.name,
            filePath: localPath,
            source: "local",
            url: `/api/jobs/${runningJob.id}/files/${encodeURIComponent(
              workflow.name,
            )}/${encodeURIComponent(filename)}`,
          });
        }
        completed = true;
      }
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
}
