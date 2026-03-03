import { NextResponse } from "next/server";
import OpenAI from "openai";

import { OPENAI_API_KEY } from "@/lib/constants";
import type { OddCatalog, OddDimension } from "@/lib/oddCatalog";

const SYSTEM_PROMPT = `You are an expert in operational design domains (ODD) for perception and safety systems.

Given a domain description, generate a factor catalog structured along 4 ODD dimensions:
1. **Actors** — objects, people, body parts, or items the system must detect or interact with
2. **Activities** — actions, movements, or behaviors relevant to the domain
3. **Environment** — environmental conditions, settings, or contextual factors
4. **Sensors** — image artifacts, sensor limitations, or capture conditions

For each dimension, generate 6–10 concise factors (short phrases, 1–5 words each). Each factor should represent a distinct, realistic variation that could affect perception system robustness.

Return valid JSON matching this exact schema:
{
  "dimensions": [
    { "key": "actors", "label": "Actors", "factors": [{ "id": "actors-0", "label": "bare hand" }, ...] },
    { "key": "activities", "label": "Activities", "factors": [{ "id": "activities-0", "label": "reaching toward blade" }, ...] },
    { "key": "environment", "label": "Environment", "factors": [{ "id": "environment-0", "label": "heavy sawdust" }, ...] },
    { "key": "sensors", "label": "Sensors", "factors": [{ "id": "sensors-0", "label": "motion blur" }, ...] }
  ]
}

Factor IDs must follow the pattern: "<dimension-key>-<index>" (e.g., "actors-0", "environment-3").`;

export async function POST(request: Request) {
  let body: { domain?: string; apiKey?: string };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const domain = (body.domain ?? "").trim();
  if (!domain) {
    return NextResponse.json(
      { error: "A domain description is required." },
      { status: 400 },
    );
  }

  const apiKey = (body.apiKey ?? "").trim() || OPENAI_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      {
        error:
          "OpenAI API key is required. Provide it in the UI or set the OPENAI_API_KEY environment variable.",
      },
      { status: 400 },
    );
  }

  try {
    const client = new OpenAI({ apiKey });

    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini",
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        {
          role: "user",
          content: `Domain: ${domain}`,
        },
      ],
      temperature: 0.7,
    });

    const raw = completion.choices[0]?.message?.content;
    if (!raw) {
      return NextResponse.json(
        { error: "OpenAI returned an empty response." },
        { status: 500 },
      );
    }

    const parsed = JSON.parse(raw) as { dimensions?: OddDimension[] };
    if (!Array.isArray(parsed.dimensions) || parsed.dimensions.length === 0) {
      return NextResponse.json(
        { error: "OpenAI returned an unexpected format." },
        { status: 500 },
      );
    }

    const catalog: OddCatalog = {
      domain,
      dimensions: parsed.dimensions,
    };

    return NextResponse.json(catalog);
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unknown error occurred.";
    return NextResponse.json(
      { error: `Failed to generate factors: ${message}` },
      { status: 500 },
    );
  }
}
