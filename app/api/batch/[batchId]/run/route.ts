import { NextResponse } from "next/server";

import { readBatch, writeBatch, runBatch } from "@/lib/batchRunner";

/**
 * POST /api/batch/[batchId]/run
 * Kicks off batch processing. Must be called after all uploads are done.
 */
export async function POST(
  _request: Request,
  { params }: { params: Promise<{ batchId: string }> },
) {
  const { batchId } = await params;
  const batch = await readBatch(batchId);
  if (!batch) {
    return NextResponse.json({ error: "Batch not found" }, { status: 404 });
  }
  if (batch.status === "running" || batch.status === "completed") {
    return NextResponse.json(
      { error: `Batch is already ${batch.status}` },
      { status: 400 },
    );
  }
  if (batch.subJobs.length === 0) {
    return NextResponse.json({ error: "No images uploaded yet" }, { status: 400 });
  }

  // Mark as running immediately so the response is fast
  await writeBatch({ ...batch, status: "running", startedAt: new Date().toISOString() });

  // Spawn async — don't await
  void runBatch(batchId).catch(async (err) => {
    const latest = await readBatch(batchId);
    if (latest) {
      await writeBatch({
        ...latest,
        status: "failed",
        error: (err as Error).message,
        completedAt: new Date().toISOString(),
      });
    }
  });

  return NextResponse.json({ status: "running", total: batch.subJobs.length });
}
