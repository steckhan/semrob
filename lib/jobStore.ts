import fs from "fs/promises";
import path from "path";

import { DATA_ROOT } from "./constants";
import type { JobRecord, JobStatus } from "./types";

const JOBS_DIR = path.join(DATA_ROOT, "jobs");

export async function ensureJobStore(): Promise<void> {
  await fs.mkdir(JOBS_DIR, { recursive: true });
}

export function jobPath(jobId: string): string {
  return path.join(JOBS_DIR, `${jobId}.json`);
}

export async function readJob(jobId: string): Promise<JobRecord | null> {
  try {
    const payload = await fs.readFile(jobPath(jobId), "utf8");
    return JSON.parse(payload) as JobRecord;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return null;
    }
    throw error;
  }
}

export async function writeJob(job: JobRecord): Promise<void> {
  await ensureJobStore();
  await fs.writeFile(jobPath(job.id), JSON.stringify(job, null, 2), "utf8");
}

export async function updateJobStatus(
  job: JobRecord,
  status: JobStatus,
  error?: string,
): Promise<JobRecord> {
  const now = new Date().toISOString();
  const updated: JobRecord = {
    ...job,
    status,
    error: error ?? job.error,
    startedAt: status === "running" ? (job.startedAt ?? now) : job.startedAt,
    completedAt:
      status === "completed" || status === "failed" ? now : job.completedAt,
  };
  await writeJob(updated);
  return updated;
}
