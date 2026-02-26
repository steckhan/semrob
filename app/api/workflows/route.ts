import { NextResponse } from "next/server";

import { writeMappings } from "@/lib/workflowMappingStore";
import { loadWorkflowBundle } from "@/lib/workflowLoader";
import { buildWorkflowMappings, normalizeMappingPayload } from "@/lib/workflowMapper";
import type { WorkflowMapping } from "@/lib/types";

export async function GET() {
  const { workflows, mappings: stored } = await loadWorkflowBundle();
  const mappings = buildWorkflowMappings({
    workflows: workflows.map((workflow) => workflow.name),
    overrides: stored,
  });

  return NextResponse.json({ workflows: mappings });
}

export async function POST(request: Request) {
  const payload = (await request.json()) as { workflows?: WorkflowMapping[] };
  if (!payload.workflows || payload.workflows.length === 0) {
    return NextResponse.json({ error: "No workflows provided." }, { status: 400 });
  }

  const normalized = normalizeMappingPayload(payload.workflows);
  await writeMappings(normalized);

  return NextResponse.json({ workflows: normalized });
}
