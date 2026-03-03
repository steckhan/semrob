import { NextResponse } from "next/server";

import { COMFYUI_BASE_URL, OPENAI_API_KEY } from "@/lib/constants";
import { createJob } from "@/lib/jobRunner";
import type { InpaintParams } from "@/lib/types";
import { loadWorkflowBundle } from "@/lib/workflowLoader";

const DEFAULT_PARAMS: InpaintParams = {
  seed: 42,
  steps: 28,
  cfgScale: 8,
  sampler: "euler",
  scheduler: "normal",
  denoise: 1,
  maskStrength: 1,
  variationCount: 4,
  useWorkflowDefaults: false,
  positivePrompt: "wristwatch, metal casing, worn look",
  negativePrompt: "",
};

function parseComfyBaseUrl(rawValue: FormDataEntryValue | null): string | null {
  if (rawValue === null) {
    return null;
  }

  const value = String(rawValue).trim();
  if (!value) {
    return null;
  }

  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    return null;
  }

  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    return null;
  }

  return value.replace(/\/+$/, "");
}

export async function POST(request: Request) {
  const formData = await request.formData();
  const imageFile = formData.get("image") as File | null;
  const maskFile = formData.get("mask") as File | null;
  const rawComfyBaseUrl = formData.get("comfyBaseUrl");
  const comfyBaseUrl = parseComfyBaseUrl(rawComfyBaseUrl);

  const inpaintMode =
    String(formData.get("inpaintMode") ?? "local") === "api" ? "api" : "local";

  const rawOpenaiApiKey = String(formData.get("openaiApiKey") ?? "").trim();
  const openaiApiKey = rawOpenaiApiKey || OPENAI_API_KEY;

  const VALID_OPENAI_MODELS = ["gpt-image-1", "gpt-image-1.5"] as const;
  type OpenAIModel = (typeof VALID_OPENAI_MODELS)[number];
  const rawOpenaiModel = String(formData.get("openaiModel") ?? "gpt-image-1").trim();
  const openaiModel: OpenAIModel = (VALID_OPENAI_MODELS as readonly string[]).includes(rawOpenaiModel)
    ? (rawOpenaiModel as OpenAIModel)
    : "gpt-image-1";

  if (!imageFile || !maskFile) {
    return NextResponse.json(
      { error: "Image and mask are required." },
      { status: 400 },
    );
  }

  if (inpaintMode === "api" && !openaiApiKey) {
    return NextResponse.json(
      {
        error:
          "OpenAI API key is required for API mode. Provide it in the UI or set the OPENAI_API_KEY environment variable.",
      },
      { status: 400 },
    );
  }

  if (
    inpaintMode === "local" &&
    rawComfyBaseUrl !== null &&
    String(rawComfyBaseUrl).trim().length > 0 &&
    comfyBaseUrl === null
  ) {
    return NextResponse.json(
      {
        error:
          "Invalid ComfyUI URL. Use a full http(s) URL like http://127.0.0.1:8188",
      },
      { status: 400 },
    );
  }

  const params: InpaintParams = {
    seed: Number(formData.get("seed") ?? DEFAULT_PARAMS.seed),
    steps: Number(formData.get("steps") ?? DEFAULT_PARAMS.steps),
    cfgScale: Number(formData.get("cfgScale") ?? DEFAULT_PARAMS.cfgScale),
    sampler: String(formData.get("sampler") ?? DEFAULT_PARAMS.sampler),
    scheduler: String(formData.get("scheduler") ?? DEFAULT_PARAMS.scheduler),
    denoise: Number(formData.get("denoise") ?? DEFAULT_PARAMS.denoise),
    maskStrength: Number(
      formData.get("maskStrength") ?? DEFAULT_PARAMS.maskStrength,
    ),
    variationCount: Number(
      formData.get("variationCount") ?? DEFAULT_PARAMS.variationCount,
    ),
    useWorkflowDefaults:
      String(formData.get("useWorkflowDefaults") ?? "false") === "true",
    positivePrompt: String(
      formData.get("positivePrompt") ?? DEFAULT_PARAMS.positivePrompt,
    ),
    negativePrompt: String(
      formData.get("negativePrompt") ?? DEFAULT_PARAMS.negativePrompt,
    ),
  };

  const { workflows, mappings } = await loadWorkflowBundle();
  if (inpaintMode === "local" && workflows.length === 0) {
    return NextResponse.json(
      { error: "No workflows found in /workflows." },
      { status: 500 },
    );
  }

  const [imageBuffer, maskBuffer] = await Promise.all([
    imageFile.arrayBuffer(),
    maskFile.arrayBuffer(),
  ]);

  const job = await createJob({
    imageBuffer: Buffer.from(imageBuffer),
    maskBuffer: Buffer.from(maskBuffer),
    comfyBaseUrl: inpaintMode === "local" ? (comfyBaseUrl ?? COMFYUI_BASE_URL) : undefined,
    params,
    workflows: inpaintMode === "local" ? workflows : [],
    mappings: inpaintMode === "local" ? mappings : [],
    inpaintMode: inpaintMode as "local" | "api",
    openaiApiKey: inpaintMode === "api" ? openaiApiKey : undefined,
    openaiModel: inpaintMode === "api" ? openaiModel : undefined,
  });

  return NextResponse.json(job);
}
