import { NextResponse } from "next/server";
import OpenAI from "openai";

import { OPENAI_API_KEY } from "@/lib/constants";
import type { OddCatalog } from "@/lib/oddCatalog";

const SYSTEM_PROMPT = `You are an expert in computer vision robustness testing for safety-critical perception systems.

You are given an Operational Design Domain (ODD) description and, optionally, an ODD factor catalog that has already been generated for that domain. Your task is to generate exactly N short inpainting prompts that represent distinct, realistic semantic variations of the **primary target object** (e.g. hand covering, PPE, worn item) within this domain.

Rules:
- Each prompt must be short (2–8 words), descriptive, and ready to use directly as a diffusion inpainting prompt.
- Base the variations on the domain context and the existing ODD Actors factors when provided — cover materials, colours, styles, surface conditions, wear states.
- Each variant must be meaningfully different — no near-duplicates.
- Focus only on the object itself; do NOT include background or scene descriptions.
- Return valid JSON: { "prompts": ["...", "...", ...] } with exactly N entries.`;

function formatCatalogContext(catalog: OddCatalog): string {
  const lines: string[] = [`ODD Domain: ${catalog.domain}`, ""];
  for (const dim of catalog.dimensions) {
    const labels = dim.factors.map((f) => f.label).join(", ");
    lines.push(`${dim.label}: ${labels}`);
  }
  return lines.join("\n");
}

export async function POST(request: Request) {
  let body: { domain?: string; n?: number; apiKey?: string; catalog?: OddCatalog };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const domain = (body.domain ?? "").trim();
  if (!domain) {
    return NextResponse.json({ error: "An ODD domain description is required." }, { status: 400 });
  }

  const n = Math.min(20, Math.max(2, Math.round(body.n ?? 4)));
  const apiKey = (body.apiKey ?? "").trim() || OPENAI_API_KEY;

  if (!apiKey) {
    return NextResponse.json(
      { error: "OpenAI API key is required. Provide it in the UI or set OPENAI_API_KEY." },
      { status: 400 },
    );
  }

  // Build user message: include ODD catalog context when available
  const userContent = body.catalog
    ? `${formatCatalogContext(body.catalog)}\n\nCount: ${n}`
    : `ODD Domain: ${domain}\n\nCount: ${n}`;

  try {
    const client = new OpenAI({ apiKey });

    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini",
      response_format: { type: "json_object" },
      temperature: 0.8,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user",   content: userContent },
      ],
    });

    const raw = completion.choices[0]?.message?.content ?? "";
    const parsed = JSON.parse(raw) as { prompts?: unknown };

    if (!Array.isArray(parsed.prompts) || parsed.prompts.length === 0) {
      return NextResponse.json({ error: "OpenAI returned an unexpected format." }, { status: 500 });
    }

    const prompts = (parsed.prompts as unknown[])
      .slice(0, n)
      .map((p) => String(p).trim())
      .filter(Boolean);

    return NextResponse.json({ prompts });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: `Generation failed: ${message}` }, { status: 500 });
  }
}
