import fs from "fs/promises";
import path from "path";

import { NextResponse } from "next/server";

import { DATA_ROOT } from "@/lib/constants";
import { readJob } from "@/lib/jobStore";

const YOLO_DIR = path.join(DATA_ROOT, "yolo");

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string; path: string[] }> },
) {
  const { id, path: pathSegments } = await params;
  const job = await readJob(id);
  if (!job) {
    return NextResponse.json({ error: "Job not found." }, { status: 404 });
  }

  const filename = pathSegments.join("/");
  if (!filename) {
    return NextResponse.json({ error: "Invalid file path." }, { status: 400 });
  }

  const filePath = path.join(YOLO_DIR, id, filename);

  try {
    const buffer = await fs.readFile(filePath);
    return new NextResponse(buffer, {
      headers: { "Content-Type": "image/png" },
    });
  } catch {
    return NextResponse.json({ error: "File not found." }, { status: 404 });
  }
}
