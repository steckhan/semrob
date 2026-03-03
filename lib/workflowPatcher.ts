import type { InpaintParams, WorkflowPatchTargets } from "./types";

type WorkflowJson = Record<string, any>;

type PatchTargets = WorkflowPatchTargets;

export const DEFAULT_PATCH_TARGETS: PatchTargets = {
  imageNodeId: "image_input",
  imageInputKey: "image",
  // maskNodeId is intentionally omitted — workflows that embed the mask in
  // the image alpha channel (e.g. flux2_klein) should not set this.
  maskNodeId: "mask_input",
  maskInputKey: "image",
  paramsNodeId: "params",
};

const IMAGE_NODE_TYPES = new Set(["LoadImage", "ImageLoader", "ImageInput"]);
const MASK_NODE_TYPES = new Set(["LoadImageMask", "MaskLoader", "MaskInput"]);
const PARAMS_NODE_TYPES = new Set(["KSampler", "KSamplerAdvanced"]);
const PROMPT_NODE_TYPES = new Set(["CLIPTextEncode", "Text Multiline"]);

/** Input key used for text on prompt-like nodes */
const PROMPT_INPUT_KEY: Record<string, string> = {
  "CLIPTextEncode": "text",
  "Text Multiline": "text",
};

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

  if (!resolved.imageNodeId || !resolved.paramsNodeId) {
    throw new Error(
      "Workflow mapping missing required node IDs. Update workflow mapping.",
    );
  }

  return resolved;
}

/**
 * Patches a ComfyUI workflow JSON with the provided image path, mask path, and
 * generation parameters. Supports both classic KSampler-based workflows and
 * the split-node pattern used by Flux2 Klein (easy seed + Flux2Scheduler +
 * KSamplerSelect + Text Multiline prompt + ImpactInt mode switch).
 */
export function patchWorkflow({
  workflow,
  imagePath,
  maskPath,
  params,
  targets,
}: WorkflowPatchInput): WorkflowJson {
  const resolvedTargets = resolveWorkflowTargets(workflow, targets);
  const patched = structuredClone(workflow);

  // --- Image node ---
  const imageNode = patched[resolvedTargets.imageNodeId];
  if (!imageNode) {
    throw new Error(
      "Workflow patching failed. Ensure nodes image, mask, and params exist or configure workflow mapping.",
    );
  }
  if (!imageNode.inputs) {
    imageNode.inputs = {};
  }
  const imageKey = resolvedTargets.imageInputKey ?? "image";
  imageNode.inputs[imageKey] = imagePath;

  // --- Mask node (skipped when mask is embedded in the image alpha channel) ---
  if (resolvedTargets.maskNodeId) {
    const maskNode = patched[resolvedTargets.maskNodeId];
    if (maskNode) {
      if (!maskNode.inputs) {
        maskNode.inputs = {};
      }
      const maskKey = resolvedTargets.maskInputKey ?? "image";
      const currentMaskValue = maskNode.inputs[maskKey];
      const isLinkedMaskInput =
        Array.isArray(currentMaskValue) &&
        currentMaskValue.length === 2 &&
        (typeof currentMaskValue[0] === "string" ||
          typeof currentMaskValue[0] === "number") &&
        typeof currentMaskValue[1] === "number";

      if (isLinkedMaskInput) {
        // The mask input is wired from another node — insert a new LoadImageMask node
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
  }

  // --- Mode switch (always applied regardless of useWorkflowDefaults) ---
  if (resolvedTargets.modeSwitchNodeId) {
    const modeSwitchNode = patched[resolvedTargets.modeSwitchNodeId];
    if (modeSwitchNode?.inputs) {
      modeSwitchNode.inputs.value = params.inpaintMode === "outpaint" ? 2 : 1;
    }
  }

  if (params.useWorkflowDefaults) {
    return patched;
  }

  // --- Params node (KSampler / easy seed / generic fallback) ---
  const paramsNode = patched[resolvedTargets.paramsNodeId];
  if (!paramsNode) {
    throw new Error(
      "Workflow patching failed. Ensure nodes image, mask, and params exist or configure workflow mapping.",
    );
  }
  if (!paramsNode.inputs) {
    paramsNode.inputs = {};
  }

  const paramsClassType: string = paramsNode.class_type ?? "";

  if (paramsClassType === "easy seed") {
    // Flux2 Klein split-node pattern: seed lives in "easy seed" node
    paramsNode.inputs.seed = params.seed;
    _patchStepsNode(patched, resolvedTargets, params);
    _patchSamplerNode(patched, resolvedTargets, params);
    _patchCfgNode(patched, resolvedTargets, params);
  } else {
    // Classic KSampler / KSamplerAdvanced
    paramsNode.inputs.seed = params.seed;
    paramsNode.inputs.steps = params.steps;
    paramsNode.inputs.cfg = params.cfgScale;
    paramsNode.inputs.sampler_name = params.sampler;
  }

  // --- Positive prompt node ---
  const positivePromptNode = resolvedTargets.positivePromptNodeId
    ? patched[resolvedTargets.positivePromptNodeId]
    : undefined;
  if (positivePromptNode?.inputs) {
    const classType: string = positivePromptNode.class_type ?? "";
    const key = PROMPT_INPUT_KEY[classType] ?? "text";
    positivePromptNode.inputs[key] = params.positivePrompt;
  }

  // --- Color match node (optional) ---
  if (resolvedTargets.colorMatchNodeId) {
    const colorMatchNode = patched[resolvedTargets.colorMatchNodeId];
    if (colorMatchNode?.inputs) {
      colorMatchNode.inputs.strength = params.colorMatchStrength;
    }
  }

  return patched;
}

/**
 * Patch a dedicated steps node (e.g. Flux2Scheduler).
 * Falls back to the paramsNode if no stepsNodeId is configured.
 */
function _patchStepsNode(
  patched: WorkflowJson,
  resolvedTargets: PatchTargets,
  params: InpaintParams,
): void {
  const nodeId = resolvedTargets.stepsNodeId ?? resolvedTargets.paramsNodeId;
  const node = patched[nodeId];
  if (!node) return;
  if (!node.inputs) node.inputs = {};
  node.inputs.steps = params.steps;
}

/**
 * Patch a dedicated sampler-select node (e.g. KSamplerSelect).
 * Falls back to the paramsNode if no samplerNodeId is configured.
 */
function _patchSamplerNode(
  patched: WorkflowJson,
  resolvedTargets: PatchTargets,
  params: InpaintParams,
): void {
  const nodeId = resolvedTargets.samplerNodeId ?? resolvedTargets.paramsNodeId;
  const node = patched[nodeId];
  if (!node) return;
  if (!node.inputs) node.inputs = {};
  node.inputs.sampler_name = params.sampler;
}

/**
 * Patch a dedicated CFG guider node (e.g. CFGGuider).
 * Falls back to the paramsNode if no cfgNodeId is configured.
 */
function _patchCfgNode(
  patched: WorkflowJson,
  resolvedTargets: PatchTargets,
  params: InpaintParams,
): void {
  const nodeId = resolvedTargets.cfgNodeId ?? resolvedTargets.paramsNodeId;
  const node = patched[nodeId];
  if (!node) return;
  if (!node.inputs) node.inputs = {};
  node.inputs.cfg = params.cfgScale;
}
