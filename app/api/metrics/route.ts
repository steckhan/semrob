import { NextResponse } from "next/server";

import { getAccumulatedMetrics } from "@/lib/metricsStore";

export async function GET() {
  const metrics = await getAccumulatedMetrics();
  return NextResponse.json(metrics);
}
