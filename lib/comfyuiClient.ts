import http from "node:http";
import https from "node:https";
import path from "path";
import { randomUUID } from "crypto";

import { DATA_ROOT } from "./constants";
import type { InpaintParams, JobOutput, WorkflowPatchTargets } from "./types";
import { patchWorkflow } from "./workflowPatcher";

// A stable client_id for this server process. ComfyUI uses it to maintain
// a node-cache context between runs — keeping model weights in VRAM so
// UNETLoader / CLIPLoader / VAELoader don't reload on every prompt.
const COMFYUI_CLIENT_ID = randomUUID();

/**
 * Minimal HTTP helper using node:http / node:https to bypass Next.js's
 * instrumented fetch (which can trigger Node.js async-context assertion
 * errors when called from background tasks on Node.js 24+).
 */
function nativeRequest(
  url: string,
  options?: { method?: string; headers?: Record<string, string>; body?: string },
): Promise<{ status: number; ok: boolean; text: string }> {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const client = parsed.protocol === "https:" ? https : http;
    const reqOptions: http.RequestOptions = {
      hostname: parsed.hostname,
      port: parsed.port || (parsed.protocol === "https:" ? 443 : 80),
      path: parsed.pathname + parsed.search,
      method: options?.method ?? "GET",
      headers: options?.headers,
    };
    const req = client.request(reqOptions, (res) => {
      const chunks: Buffer[] = [];
      res.on("data", (chunk: Buffer) => chunks.push(chunk));
      res.on("end", () => {
        const statusCode = res.statusCode ?? 0;
        const text = Buffer.concat(chunks).toString("utf8");
        resolve({ status: statusCode, ok: statusCode >= 200 && statusCode < 300, text });
      });
      res.on("error", reject);
    });
    req.on("error", reject);
    if (options?.body) req.write(options.body);
    req.end();
  });
}

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
  let result: { status: number; ok: boolean; text: string };
  try {
    result = await nativeRequest(`${comfyBaseUrl}/prompt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: patched }),
    });
  } catch (error) {
    throw new Error(
      `Could not reach ComfyUI at ${comfyBaseUrl}. Check the URL and that ComfyUI is running.`,
      { cause: error },
    );
  }

  if (!result.ok) {
    throw new Error(
      `ComfyUI prompt failed with ${result.status}: ${result.text || "No response"}`,
    );
  }

  const payload = JSON.parse(result.text) as ComfyPromptResponse;
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
  let result: { status: number; ok: boolean; text: string };
  try {
    result = await nativeRequest(`${comfyBaseUrl}/history/${promptId}`);
  } catch (error) {
    throw new Error(
      `Could not reach ComfyUI at ${comfyBaseUrl} while polling history.`,
      { cause: error },
    );
  }
  if (!result.ok) {
    throw new Error(`ComfyUI history failed with ${result.status}`);
  }

  const payload = JSON.parse(result.text) as ComfyHistoryResponse;
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
  const recordOutputs = record.outputs ?? {};
  Object.values(recordOutputs).forEach((output) => {
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
    console.log(`[ComfyUI] Prompt ${promptId} completed with ${outputs.length} output(s).`);
    return { done: true, outputs };
  }

  // Record exists but no output-type images yet.
  // If status is "success" or completed=true the workflow finished without
  // producing any SaveImage output — treat as an error.
  if (statusStr === "success" || record.status?.completed === true) {
    console.warn(`[ComfyUI] Prompt ${promptId} status="${statusStr}" but no output images found. Outputs:`, JSON.stringify(recordOutputs));
    return {
      done: true,
      error: "ComfyUI workflow completed but produced no output images. Check that SaveImage nodes executed correctly.",
    };
  }

  // Still running (record present but not yet completed)
  console.log(`[ComfyUI] Prompt ${promptId} in history, status="${statusStr}", outputs keys: [${Object.keys(recordOutputs).join(",")}] — still waiting`);
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
  console.log(`[ComfyUI] Waiting for prompt ${promptId} at ${comfyBaseUrl}`);
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

  let buffer: Buffer;
  try {
    buffer = await new Promise<Buffer>((resolve, reject) => {
      const client = url.protocol === "https:" ? https : http;
      client.get(url.toString(), (res) => {
        if (res.statusCode !== 200) {
          reject(new Error(`ComfyUI view failed with ${res.statusCode}`));
          res.resume();
          return;
        }
        const chunks: Buffer[] = [];
        res.on("data", (chunk: Buffer) => chunks.push(chunk));
        res.on("end", () => resolve(Buffer.concat(chunks)));
        res.on("error", reject);
      }).on("error", reject);
    });
  } catch (error) {
    throw new Error(
      `Could not reach ComfyUI at ${comfyBaseUrl} while fetching image output.`,
      { cause: error },
    );
  }

  return buffer.buffer as ArrayBuffer;
}
