import path from "path";
import { randomUUID } from "crypto";

import { DATA_ROOT } from "./constants";
import type { InpaintParams, JobOutput, WorkflowPatchTargets } from "./types";
import { patchWorkflow } from "./workflowPatcher";

// A stable client_id for this server process. ComfyUI uses it to maintain
// a node-cache context between runs — keeping model weights in VRAM so
// UNETLoader / CLIPLoader / VAELoader don't reload on every prompt.
const COMFYUI_CLIENT_ID = randomUUID();

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
    status?: {
      status_str?: string;       // "success" | "error"
      completed?: boolean;
      messages?: Array<[string, unknown]>;
    };
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

type PollResult =
  | { done: false }
  | { done: true; outputs: JobOutput[] }
  | { done: true; error: string };

async function checkWorkflowHistory(
  promptId: string,
  comfyBaseUrl: string,
): Promise<PollResult> {
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

  // Not in history yet — still queued or running
  if (!record) {
    return { done: false };
  }

  // Check for explicit error status
  const statusStr = record.status?.status_str;
  if (statusStr === "error") {
    const msgs = record.status?.messages ?? [];
    const errMsg = msgs.find(([type]) => type === "execution_error")?.[1];
    return {
      done: true,
      error: errMsg ? JSON.stringify(errMsg) : "ComfyUI reported an execution error",
    };
  }

  // Collect type:"output" images (SaveImage nodes)
  const outputs: JobOutput[] = [];
  let variationIndex = 0;
  Object.values(record.outputs).forEach((output) => {
    output.images
      ?.filter((image) => image.type === "output")
      .forEach((image) => {
        const filePath = path.join(
          DATA_ROOT,
          "outputs",
          promptId,
          image.subfolder,
          image.filename,
        );
        outputs.push({
          workflowName: "",
          variationIndex: variationIndex++,
          filePath,
          url: "",
          source: "comfyui",
          subfolder: image.subfolder,
          filename: image.filename,
          imageType: image.type,
        });
      });
  });

  if (outputs.length > 0) {
    return { done: true, outputs };
  }

  // Record exists but no output-type images yet.
  // If status is "success" or completed=true the workflow finished without
  // producing any SaveImage output — treat as an error.
  if (statusStr === "success" || record.status?.completed === true) {
    return {
      done: true,
      error: "ComfyUI workflow completed but produced no output images. Check that SaveImage nodes executed correctly.",
    };
  }

  // Still running (record present but not yet completed)
  return { done: false };
}

export async function pollWorkflow(
  promptId: string,
  comfyBaseUrl: string,
): Promise<JobOutput[]> {
  const result = await checkWorkflowHistory(promptId, comfyBaseUrl);
  if (!result.done) return [];
  if ("error" in result) throw new Error(result.error);
  return result.outputs;
}

/**
 * Wait for a ComfyUI prompt to finish by polling /history with an adaptive
 * interval: starts at 300ms and backs off up to 1500ms.
 * Returns the outputs as soon as they appear, or throws on error/timeout.
 */
export async function waitForWorkflow(
  promptId: string,
  comfyBaseUrl: string,
  timeoutMs = 300_000,
): Promise<JobOutput[]> {
  const start = Date.now();
  let delay = 300;
  while (Date.now() - start < timeoutMs) {
    await new Promise((resolve) => setTimeout(resolve, delay));
    // Back off: 300 → 600 → 1200 → 1500ms cap
    delay = Math.min(delay * 2, 1500);

    const result = await checkWorkflowHistory(promptId, comfyBaseUrl);
    if (!result.done) continue;
    if ("error" in result) throw new Error(result.error);
    return result.outputs;
  }
  throw new Error(`Timed out waiting for ComfyUI prompt ${promptId}`);
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
