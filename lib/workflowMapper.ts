import type { WorkflowMapping, WorkflowPatchTargets } from "./types";
import { DEFAULT_PATCH_TARGETS } from "./workflowPatcher";

export type WorkflowMapInput = {
  workflows: string[];
  overrides?: WorkflowMapping[];
};

export function buildWorkflowMappings({
  workflows,
  overrides,
}: WorkflowMapInput): WorkflowMapping[] {
  const overrideMap = new Map(
    (overrides ?? []).map((entry) => [entry.workflowName, entry.targets]),
  );

  return workflows.map((name) => ({
    workflowName: name,
    targets: {
      ...DEFAULT_PATCH_TARGETS,
      ...(overrideMap.get(name) ?? {}),
    },
  }));
}

export function normalizeMappingPayload(
  payload: WorkflowMapping[],
): WorkflowMapping[] {
  return payload.map((entry) => ({
    workflowName: entry.workflowName,
    targets: {
      imageNodeId: entry.targets.imageNodeId,
      imageInputKey: entry.targets.imageInputKey ?? "image",
      maskNodeId: entry.targets.maskNodeId,
      maskInputKey: entry.targets.maskInputKey ?? "image",
      paramsNodeId: entry.targets.paramsNodeId,
      positivePromptNodeId: entry.targets.positivePromptNodeId,
      negativePromptNodeId: entry.targets.negativePromptNodeId,
    },
  }));
}
