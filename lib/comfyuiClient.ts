import path from "path";

import { DATA_ROOT } from "./constants";
import type { InpaintParams, JobOutput, WorkflowPatchTargets } from "./types";
import { patchWorkflow } from "./workflowPatcher";

export type ComfySubmitResult = {
  promptId: string;
};

export type ComfyWorkflowRun = {
  workflowName: string;
  workflowJson: Record<string, unknown>;
  targets: WorkflowPatchTargets;
};

type ComfyPromptResponse = {
  prompt_id: string;
};

type ComfyHistoryResponse = {
  [promptId: string]: {
    outputs: Record<
      string,
      {
        images?: Array<{
          filename: string;
          subfolder: string;
          type: string;
        }>;
      }
    >;
  };
};

export function buildPatchedWorkflow(
  workflowRun: ComfyWorkflowRun,
  imagePath: string,
  maskPath: string,
  params: InpaintParams,
): Record<string, unknown> {
  return patchWorkflow({
    workflow: workflowRun.workflowJson,
    imagePath,
    maskPath,
    params,
    targets: workflowRun.targets,
  });
}

export async function submitPatchedWorkflow(
  patched: Record<string, unknown>,
  comfyBaseUrl: string,
): Promise<ComfySubmitResult> {
  let response: Response;
  try {
    response = await fetch(`${comfyBaseUrl}/prompt`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ prompt: patched }),
    });
  } catch (error) {
    throw new Error(
      `Could not reach ComfyUI at ${comfyBaseUrl}. Check the URL and that ComfyUI is running.`,
      { cause: error },
    );
  }

  if (!response.ok) {
    const body = await response.text();
    throw new Error(
      `ComfyUI prompt failed with ${response.status}: ${body || "No response"}`,
    );
  }

  const payload = (await response.json()) as ComfyPromptResponse;
  return { promptId: payload.prompt_id };
}

export async function pollWorkflow(
  promptId: string,
  comfyBaseUrl: string,
): Promise<JobOutput[]> {
  let response: Response;
  try {
    response = await fetch(`${comfyBaseUrl}/history/${promptId}`);
  } catch (error) {
    throw new Error(
      `Could not reach ComfyUI at ${comfyBaseUrl} while polling history.`,
      { cause: error },
    );
  }
  if (!response.ok) {
    throw new Error(`ComfyUI history failed with ${response.status}`);
  }

  const payload = (await response.json()) as ComfyHistoryResponse;
  const record = payload[promptId];
  if (!record) {
    return [];
  }

  const outputs: JobOutput[] = [];
  let outputIndex = 0;
  Object.values(record.outputs).forEach((output) => {
    output.images?.forEach((image) => {
      // Skip temp/preview images — only collect saved output images
      if (image.type === "temp") {
        return;
      }
      const filePath = path.join(
        DATA_ROOT,
        "outputs",
        promptId,
        image.subfolder,
        image.filename,
      );
      outputs.push({
        workflowName: "",
        variationIndex: outputIndex,
        filePath,
        url: "",
        source: "comfyui",
        subfolder: image.subfolder,
        filename: image.filename,
        imageType: image.type,
      });
      outputIndex += 1;
    });
  });

  return outputs;
}

export async function fetchImageBuffer(
  imagePath: string,
  subfolder: string,
  comfyBaseUrl: string,
): Promise<ArrayBuffer> {
  const url = new URL(`${comfyBaseUrl}/view`);
  url.searchParams.set("filename", imagePath);
  url.searchParams.set("subfolder", subfolder);
  url.searchParams.set("type", "output");

  let response: Response;
  try {
    response = await fetch(url.toString());
  } catch (error) {
    throw new Error(
      `Could not reach ComfyUI at ${comfyBaseUrl} while fetching image output.`,
      { cause: error },
    );
  }
  if (!response.ok) {
    throw new Error(`ComfyUI view failed with ${response.status}`);
  }

  return response.arrayBuffer();
}
