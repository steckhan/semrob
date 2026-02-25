import { NextResponse } from "next/server";

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

export async function POST(request: Request) {
  const formData = await request.formData();
  const imageFile = formData.get("image") as File | null;
  const maskFile = formData.get("mask") as File | null;

  if (!imageFile || !maskFile) {
    return NextResponse.json(
      { error: "Image and mask are required." },
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
  if (workflows.length === 0) {
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
    params,
    workflows,
    mappings,
  });

  return NextResponse.json(job);
}
