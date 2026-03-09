import fs from "fs/promises";
import path from "path";

import type { WorkflowMapping } from "./types";
import { WORKFLOWS_DIR } from "./constants";

const MAPPING_FILE = path.join(WORKFLOWS_DIR, "workflow-mapping.json");

export async function readMappings(): Promise<WorkflowMapping[]> {
  try {
    const raw = await fs.readFile(MAPPING_FILE, "utf8");
    if (!raw.trim()) return [];
    const json = JSON.parse(raw) as WorkflowMapping[];
    return json;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return [];
    }
    throw error;
  }
}

export async function writeMappings(mappings: WorkflowMapping[]): Promise<void> {
  await fs.mkdir(WORKFLOWS_DIR, { recursive: true });
  await fs.writeFile(MAPPING_FILE, JSON.stringify(mappings, null, 2), "utf8");
}
