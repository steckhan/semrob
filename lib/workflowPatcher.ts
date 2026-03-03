import type { InpaintParams, WorkflowPatchTargets } from "./types";

type WorkflowJson = Record<string, any>;

type PatchTargets = WorkflowPatchTargets;

export const DEFAULT_PATCH_TARGETS: PatchTargets = {
  imageNodeId: "image_input",
  imageInputKey: "image",
  maskNodeId: "mask_input",
  maskInputKey: "image",
  paramsNodeId: "params",
};

const IMAGE_NODE_TYPES = new Set(["LoadImage", "ImageLoader", "ImageInput"]);
const MASK_NODE_TYPES = new Set(["LoadImageMask", "MaskLoader", "MaskInput"]);
const PARAMS_NODE_TYPES = new Set(["KSampler", "KSamplerAdvanced"]);
const PROMPT_NODE_TYPES = new Set(["CLIPTextEncode"]);

function inferTargets(workflow: WorkflowJson): PatchTargets | null {
  let imageNodeId: string | undefined;
  let maskNodeId: string | undefined;
  let paramsNodeId: string | undefined;
  const promptNodes: string[] = [];

  Object.entries(workflow).forEach(([nodeId, node]) => {
    if (!node || typeof node !== "object") {
      return;
    }
    const classType = (node as { class_type?: string }).class_type;
    if (!classType) {
      return;
    }
    if (!imageNodeId && IMAGE_NODE_TYPES.has(classType)) {
      imageNodeId = nodeId;
    }
    if (!maskNodeId && MASK_NODE_TYPES.has(classType)) {
      maskNodeId = nodeId;
    }
    if (!paramsNodeId && PARAMS_NODE_TYPES.has(classType)) {
      paramsNodeId = nodeId;
    }
    if (PROMPT_NODE_TYPES.has(classType)) {
      promptNodes.push(nodeId);
    }
  });

  if (!imageNodeId || !maskNodeId || !paramsNodeId) {
    return null;
  }

  return {
    imageNodeId,
    imageInputKey: "image",
    maskNodeId,
    maskInputKey: "image",
    paramsNodeId,
    positivePromptNodeId: promptNodes[0],
    negativePromptNodeId: promptNodes[1],
  };
}

function patchParam(
  patched: WorkflowJson,
  targets: PatchTargets,
  nodeIdKey: keyof PatchTargets,
  inputKeyKey: keyof PatchTargets,
  defaultInputKey: string,
  value: unknown,
): void {
  const nodeId = (targets[nodeIdKey] as string | undefined) ?? targets.paramsNodeId;
  const inputKey = (targets[inputKeyKey] as string | undefined) ?? defaultInputKey;
  const node = patched[nodeId];
  if (node?.inputs) {
    node.inputs[inputKey] = value;
  }
}

export type WorkflowPatchInput = {
  workflow: WorkflowJson;
  imagePath: string;
  maskPath: string;
  params: InpaintParams;
  targets?: Partial<PatchTargets>;
};

export function resolveWorkflowTargets(
  workflow: WorkflowJson,
  overrides?: Partial<PatchTargets>,
): PatchTargets {
  const inferred = inferTargets(workflow);
  const base = inferred ?? DEFAULT_PATCH_TARGETS;
  const resolved = { ...base, ...overrides };

  if (!resolved.imageNodeId || !resolved.maskNodeId || !resolved.paramsNodeId) {
    throw new Error(
      "Workflow mapping missing required node IDs. Update workflow mapping.",
    );
  }

  return resolved;
}

export function patchWorkflow({
  workflow,
  imagePath,
  maskPath,
  params,
  targets,
}: WorkflowPatchInput): WorkflowJson {
  const resolvedTargets = resolveWorkflowTargets(workflow, targets);
  const patched = structuredClone(workflow);

  const imageNode = patched[resolvedTargets.imageNodeId];
  const maskNode = patched[resolvedTargets.maskNodeId];
  const paramsNode = patched[resolvedTargets.paramsNodeId];
  const positivePromptNode = resolvedTargets.positivePromptNodeId
    ? patched[resolvedTargets.positivePromptNodeId]
    : undefined;
  const negativePromptNode = resolvedTargets.negativePromptNodeId
    ? patched[resolvedTargets.negativePromptNodeId]
    : undefined;

  if (!imageNode || !maskNode || !paramsNode) {
    throw new Error(
      "Workflow patching failed. Ensure nodes image, mask, and params exist or configure workflow mapping.",
    );
  }

  const imageKey = resolvedTargets.imageInputKey ?? "image";
  const maskKey = resolvedTargets.maskInputKey ?? "image";

  if (!imageNode.inputs) {
    imageNode.inputs = {};
  }
  if (!maskNode.inputs) {
    maskNode.inputs = {};
  }

  imageNode.inputs[imageKey] = imagePath;

  // If image and mask map to the same node+key, the mask is embedded as the
  // alpha channel of the image (e.g. flux2_klein's single LoadImage node).
  // jobRunner already composited the mask into the image alpha, so skip the
  // mask write here to avoid overwriting the image path with the mask path.
  const isSameNodeAndKey =
    resolvedTargets.maskNodeId === resolvedTargets.imageNodeId &&
    maskKey === imageKey;

  if (!isSameNodeAndKey) {
    const currentMaskValue = maskNode.inputs[maskKey];
    const isLinkedMaskInput =
      Array.isArray(currentMaskValue) &&
      currentMaskValue.length === 2 &&
      (typeof currentMaskValue[0] === "string" ||
        typeof currentMaskValue[0] === "number") &&
      typeof currentMaskValue[1] === "number";

    if (isLinkedMaskInput) {
      const numericIds = Object.keys(patched)
        .map((id) => Number(id))
        .filter((id) => Number.isFinite(id));
      const nextNodeId = String((Math.max(0, ...numericIds) + 1));

      patched[nextNodeId] = {
        inputs: {
          image: maskPath,
          channel: "red",
        },
        class_type: "LoadImageMask",
        _meta: {
          title: "Load Image (Mask)",
        },
      };

      maskNode.inputs[maskKey] = [nextNodeId, 0];
    } else {
      maskNode.inputs[maskKey] = maskPath;
    }
  }

  if (!paramsNode.inputs) {
    paramsNode.inputs = {};
  }

  if (!params.useWorkflowDefaults) {
    patchParam(patched, resolvedTargets, "seedNodeId", "seedInputKey", "seed", params.seed);
    patchParam(patched, resolvedTargets, "stepsNodeId", "stepsInputKey", "steps", params.steps);
    patchParam(patched, resolvedTargets, "cfgNodeId", "cfgInputKey", "cfg", params.cfgScale);
    patchParam(patched, resolvedTargets, "samplerNameNodeId", "samplerNameInputKey", "sampler_name", params.sampler);
    patchParam(patched, resolvedTargets, "schedulerNodeId", "schedulerInputKey", "scheduler", params.scheduler);
    patchParam(patched, resolvedTargets, "denoiseNodeId", "denoiseInputKey", "denoise", params.denoise);

    if (positivePromptNode?.inputs) {
      positivePromptNode.inputs.text = params.positivePrompt;
    }
    if (negativePromptNode?.inputs) {
      negativePromptNode.inputs.text = params.negativePrompt;
    }
  }

  return patched;
}
