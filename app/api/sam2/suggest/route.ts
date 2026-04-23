import { NextResponse } from "next/server";
import OpenAI from "openai";

import { OPENAI_API_KEY } from "@/lib/constants";

const SYSTEM_PROMPT = `You are an expert in computer vision and image segmentation for safety-critical systems.

Given an Operational Design Domain (ODD) description, identify the single primary physical object that should be segmented/masked for robustness testing.

Rules:
- Reply with 1–4 words only — no punctuation, no explanation.
- Output exactly the segmentation target label (e.g. "hand", "safety glove", "helmet", "face").
- Focus on the foreground object being manipulated or worn, not the background scene.
- Return valid JSON: { "target": "..." }`;

export async function POST(request: Request) {
  let body: { domain?: string; apiKey?: string };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const domain = (body.domain ?? "").trim();
  if (!domain) {
    return NextResponse.json({ error: "An ODD domain description is required." }, { status: 400 });
  }

  const apiKey = (body.apiKey ?? "").trim() || OPENAI_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { error: "OpenAI API key is required. Provide it in the UI or set OPENAI_API_KEY." },
      { status: 400 },
    );
  }

  try {
    const client = new OpenAI({ apiKey });

    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini",
      response_format: { type: "json_object" },
      temperature: 0.2,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: `ODD Domain: ${domain}` },
      ],
    });

    const raw = completion.choices[0]?.message?.content ?? "";
    const parsed = JSON.parse(raw) as { target?: unknown };

    if (typeof parsed.target !== "string" || !parsed.target.trim()) {
      return NextResponse.json({ error: "Model returned an unexpected format." }, { status: 500 });
    }

    return NextResponse.json({ target: parsed.target.trim() });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: `Suggestion failed: ${message}` }, { status: 500 });
  }
}
