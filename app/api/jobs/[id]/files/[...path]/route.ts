import fs from "fs/promises";
import path from "path";

import { NextResponse } from "next/server";

import { COMFYUI_BASE_URL } from "@/lib/constants";
import { readJob } from "@/lib/jobStore";

export async function GET(
  _request: Request,
  { params }: { params: { id: string; path: string[] } },
) {
  const job = await readJob(params.id);
  if (!job) {
    return NextResponse.json({ error: "Job not found." }, { status: 404 });
  }

  const [workflowName, filename] = params.path;
  if (!workflowName || !filename) {
    return NextResponse.json({ error: "Invalid file path." }, { status: 400 });
  }

  const match = job.outputs.find(
    (output) =>
      output.workflowName === workflowName &&
      (output.filename ?? path.basename(output.filePath)) === filename,
  );

  if (!match) {
    return NextResponse.json({ error: "File not found." }, { status: 404 });
  }

  if (match.source === "comfyui" && match.filename) {
    const url = new URL(`${COMFYUI_BASE_URL}/view`);
    url.searchParams.set("filename", match.filename);
    if (match.subfolder) {
      url.searchParams.set("subfolder", match.subfolder);
    }
    url.searchParams.set("type", match.imageType ?? "output");

    const response = await fetch(url.toString());
    if (!response.ok) {
      return NextResponse.json(
        { error: "Failed to fetch output from ComfyUI." },
        { status: response.status },
      );
    }

    const buffer = await response.arrayBuffer();
    return new NextResponse(Buffer.from(buffer), {
      headers: { "Content-Type": "image/png" },
    });
  }

  const fileBuffer = await fs.readFile(match.filePath);
  return new NextResponse(fileBuffer, {
    headers: { "Content-Type": "image/png" },
  });
}
