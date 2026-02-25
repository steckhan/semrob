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
import { DEFAULT_PATCH_TARGETS } from "./workflowPatcher";

const UPLOADS_DIR = path.join(DATA_ROOT, "uploads");
const OUTPUTS_DIR = path.join(DATA_ROOT, "outputs");

async function findComfyOutputPath(
  promptId: string,
  filename: string,
  subfolder: string | undefined,
  imageType: string | undefined,
): Promise<string> {
  if (subfolder && subfolder.length > 0) {
    return `${COMFYUI_BASE_URL}/view?filename=${encodeURIComponent(
      filename,
    )}&subfolder=${encodeURIComponent(subfolder)}&type=${encodeURIComponent(
      imageType ?? "output",
    )}`;
  }

  const response = await fetch(`${COMFYUI_BASE_URL}/history/${promptId}`);
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

  return `${COMFYUI_BASE_URL}/view?filename=${encodeURIComponent(
    filename,
  )}&subfolder=${encodeURIComponent(subfolderValue)}&type=${encodeURIComponent(
    imageType ?? "output",
  )}`;
}

async function copyOutputToLocal(
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
};

export async function createJob({
  imageBuffer,
  maskBuffer,
  params,
  workflows,
  mappings,
}: CreateJobInput): Promise<JobRecord> {
  await ensureJobStore();

  const jobId = randomUUID();
  const jobDir = path.join(UPLOADS_DIR, jobId);
  await fs.mkdir(jobDir, { recursive: true });

  const imagePath = path.join(jobDir, "input.png");
  const maskPath = path.join(jobDir, "mask.png");

  await fs.writeFile(imagePath, imageBuffer);
  await fs.writeFile(maskPath, maskBuffer);

  const comfyJobDir = path.join(COMFYUI_INPUT_DIR, jobId);
  const comfyImagePath = path.join(comfyJobDir, "input.png");
  const comfyMaskPath = path.join(comfyJobDir, "mask.png");

  await fs.mkdir(comfyJobDir, { recursive: true });
  await fs.writeFile(comfyImagePath, imageBuffer);
  await fs.writeFile(comfyMaskPath, maskBuffer);

  const job: JobRecord = {
    id: jobId,
    createdAt: new Date().toISOString(),
    status: "queued",
    params,
    workflows: workflows.map((workflow) => workflow.name),
    promptIds: {},
    outputs: [],
  };

  await writeJob(job);
  const comfyImagePathWindows = path
    .join(COMFYUI_INPUT_DIR_WINDOWS, jobId, "input.png")
    .replace(/\\/g, "/");
  const comfyMaskPathWindows = path
    .join(COMFYUI_INPUT_DIR_WINDOWS, jobId, "mask.png")
    .replace(/\\/g, "/");

  void runJob(
    job,
    workflows,
    mappings,
    comfyImagePathWindows,
    comfyMaskPathWindows,
  ).catch(async (error) => {
    await updateJobStatus(job, "failed", (error as Error).message);
  });

  return job;
}

async function runJob(
  job: JobRecord,
  workflows: WorkflowDefinition[],
  mappings: WorkflowMapping[],
  imagePath: string,
  maskPath: string,
): Promise<void> {
  await updateJobStatus(job, "running");
  await fs.mkdir(OUTPUTS_DIR, { recursive: true });

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
      job.params,
    );
    const { promptId } = await submitPatchedWorkflow(patchedWorkflow);
    promptIds[workflow.name] = promptId;
    const patchedWorkflows = {
      ...(job.patchedWorkflows ?? {}),
      [workflow.name]: patchedWorkflow,
    };
    await writeJob({ ...job, promptIds, patchedWorkflows });

    let completed = false;
    while (!completed) {
      await new Promise((resolve) => setTimeout(resolve, 1500));
      const pollOutputs = await pollWorkflow(promptId);
      if (pollOutputs.length > 0) {
        for (const output of pollOutputs) {
          const filename = output.filename ?? path.basename(output.filePath);
          const subfolder = output.subfolder ?? "";
          const localDir = path.join(OUTPUTS_DIR, job.id, workflow.name);
          const localPath = path.join(localDir, filename);
          await fs.mkdir(localDir, { recursive: true });
          await copyOutputToLocal(
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
            url: `/api/jobs/${job.id}/files/${encodeURIComponent(
              workflow.name,
            )}/${encodeURIComponent(filename)}`,
          });
        }
        completed = true;
      }
    }
  });

  await updateJobStatus(
    {
      ...job,
      outputs,
      promptIds,
    },
    "completed",
  );
}
