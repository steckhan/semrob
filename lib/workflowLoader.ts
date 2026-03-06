import fs from "fs/promises";
import path from "path";

import { WORKFLOWS_DIR } from "./constants";
import type { WorkflowDefinition, WorkflowMapping } from "./types";
import { readMappings } from "./workflowMappingStore";

type ComfyGraphLink = [number, number, number, number, number, string];

type ComfyGraphInput = {
  name: string;
  type?: string;
  link: number | null;
  widget?: {
    name: string;
  };
};

type ComfyGraphNode = {
  id: number;
  type: string;
  title?: string;
  inputs?: ComfyGraphInput[];
  widgets_values?: unknown[];
};

type ComfyGraphWorkflow = {
  nodes: ComfyGraphNode[];
  links?: ComfyGraphLink[];
};

export type WorkflowBundle = {
  workflows: WorkflowDefinition[];
  mappings: WorkflowMapping[];
};

function isComfyGraphWorkflow(value: unknown): value is ComfyGraphWorkflow {
  if (!value || typeof value !== "object") {
    return false;
  }

  return Array.isArray((value as { nodes?: unknown[] }).nodes);
}

function isValueCompatibleWithInputType(
  inputType: string | undefined,
  value: unknown,
): boolean {
  if (!inputType) {
    return true;
  }

  if (inputType === "INT" || inputType === "FLOAT") {
    return typeof value === "number";
  }

  if (inputType === "BOOLEAN") {
    return typeof value === "boolean";
  }

  if (inputType === "STRING" || inputType === "COMBO") {
    return typeof value === "string" || typeof value === "number";
  }

  return true;
}

function convertComfyGraphToPrompt(
  graph: ComfyGraphWorkflow,
): Record<string, unknown> {
  const linkMap = new Map<number, { sourceNodeId: number; sourceSlot: number }>();
  for (const link of graph.links ?? []) {
    const [linkId, sourceNodeId, sourceSlot] = link;
    linkMap.set(linkId, { sourceNodeId, sourceSlot });
  }

  const prompt: Record<string, unknown> = {};

  for (const node of graph.nodes) {
    const nodeInputs: Record<string, unknown> = {};
    const inputDefs = node.inputs ?? [];
    const widgetValues = node.widgets_values ?? [];
    let widgetIndex = 0;

    for (const input of inputDefs) {
      if (input.link !== null) {
        const linked = linkMap.get(input.link);
        if (linked) {
          nodeInputs[input.name] = [String(linked.sourceNodeId), linked.sourceSlot];
        }
        // If this linked input also has a widget, ComfyUI still stores a
        // widgets_values slot for it (the saved/default value). Advance the
        // index so subsequent widget inputs read the correct slot.
        if (input.widget) {
          widgetIndex += 1;
        }
        continue;
      }

      if (!input.widget) {
        continue;
      }

      while (
        widgetIndex < widgetValues.length &&
        !isValueCompatibleWithInputType(input.type, widgetValues[widgetIndex])
      ) {
        widgetIndex += 1;
      }

      if (widgetIndex >= widgetValues.length) {
        continue;
      }

      nodeInputs[input.name] = widgetValues[widgetIndex];
      widgetIndex += 1;
    }

    prompt[String(node.id)] = {
      inputs: nodeInputs,
      class_type: node.type,
      _meta: {
        title: node.title ?? node.type,
      },
    };
  }

  return prompt;
}

export async function loadWorkflows(): Promise<WorkflowDefinition[]> {
  const entries = await fs.readdir(WORKFLOWS_DIR, { withFileTypes: true });
  const workflows = await Promise.all(
    entries
      .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
      .filter((entry) => entry.name !== "workflow-mapping.json")
      .map(async (entry) => {
        const filePath = path.join(WORKFLOWS_DIR, entry.name);
        const raw = await fs.readFile(filePath, "utf8");
        if (!raw.trim()) return null;
        const parsed = JSON.parse(raw) as unknown;
        const json = isComfyGraphWorkflow(parsed)
          ? convertComfyGraphToPrompt(parsed)
          : (parsed as Record<string, unknown>);
        return {
          name: entry.name.replace(/\.json$/, ""),
          filePath,
          json,
        };
      }),
  );

  return workflows
    .filter((w): w is WorkflowDefinition => w !== null)
    .sort((a, b) => a.name.localeCompare(b.name));
}

export async function loadWorkflowBundle(): Promise<WorkflowBundle> {
  const [workflows, mappings] = await Promise.all([
    loadWorkflows(),
    readMappings(),
  ]);

  if (mappings.length === 0) {
    return { workflows, mappings };
  }

  const mappedNames = new Set(mappings.map((mapping) => mapping.workflowName));
  const mappedWorkflows = workflows.filter((workflow) =>
    mappedNames.has(workflow.name),
  );

  return {
    workflows: mappedWorkflows.length > 0 ? mappedWorkflows : workflows,
    mappings,
  };
}
