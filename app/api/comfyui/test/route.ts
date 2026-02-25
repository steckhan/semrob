import { NextResponse } from "next/server";

import { COMFYUI_BASE_URL } from "../../../../lib/constants";

function parseComfyBaseUrl(value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }

  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    return null;
  }

  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    return null;
  }

  return trimmed.replace(/\/+$/, "");
}

export async function POST(request: Request) {
  const payload = (await request.json()) as { comfyBaseUrl?: string };
  const targetUrl = payload.comfyBaseUrl
    ? parseComfyBaseUrl(payload.comfyBaseUrl)
    : COMFYUI_BASE_URL;

  if (!targetUrl) {
    return NextResponse.json(
      {
        ok: false,
        error:
          "Invalid ComfyUI URL. Use a full http(s) URL like http://127.0.0.1:8188",
      },
      { status: 400 },
    );
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);

  try {
    const response = await fetch(`${targetUrl}/system_stats`, {
      signal: controller.signal,
    });

    if (!response.ok) {
      return NextResponse.json(
        {
          ok: false,
          error: `ComfyUI responded with status ${response.status}.`,
        },
        { status: 502 },
      );
    }

    return NextResponse.json({ ok: true, comfyBaseUrl: targetUrl });
  } catch {
    return NextResponse.json(
      {
        ok: false,
        error: `Could not connect to ComfyUI at ${targetUrl}`,
      },
      { status: 502 },
    );
  } finally {
    clearTimeout(timeout);
  }
}
