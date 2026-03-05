import { NextResponse } from "next/server";

import { readBatch } from "@/lib/batchRunner";

/** GET /api/batch/[batchId] — poll batch status */
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ batchId: string }> },
) {
  const { batchId } = await params;
  const batch = await readBatch(batchId);
  if (!batch) {
    return NextResponse.json({ error: "Batch not found" }, { status: 404 });
  }
  return NextResponse.json(batch);
}
