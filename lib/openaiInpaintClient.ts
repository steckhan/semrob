import OpenAI, { toFile } from "openai";
import sharp from "sharp";

export type OpenAIInpaintInput = {
  imageBuffer: Buffer;
  maskBuffer: Buffer;
  prompt: string;
  apiKey: string;
  n?: number;
  model?: string;
};

/**
 * Converts our mask format (white = inpaint area, black = keep area)
 * to the RGBA format OpenAI expects (alpha=0 = inpaint area, alpha=255 = keep area).
 */
async function convertMaskForOpenAI(maskBuffer: Buffer): Promise<Buffer> {
  const { data: rawData, info } = await sharp(maskBuffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const rgba = Buffer.from(rawData);
  for (let i = 0; i < rgba.length; i += 4) {
    const isInpaintArea = rgba[i] > 128; // R channel: white = inpaint
    rgba[i] = 0;
    rgba[i + 1] = 0;
    rgba[i + 2] = 0;
    rgba[i + 3] = isInpaintArea ? 0 : 255; // transparent = edit, opaque = keep
  }

  return sharp(rgba, {
    raw: { width: info.width, height: info.height, channels: 4 },
  })
    .png()
    .toBuffer();
}

export async function runOpenAIInpainting({
  imageBuffer,
  maskBuffer,
  prompt,
  apiKey,
  n = 1,
  model = "gpt-image-1",
}: OpenAIInpaintInput): Promise<Buffer[]> {
  const [pngImage, convertedMask] = await Promise.all([
    sharp(imageBuffer).png().toBuffer(),
    convertMaskForOpenAI(maskBuffer),
  ]);

  const client = new OpenAI({ apiKey });

  const response = await client.images.edit({
    model,
    image: await toFile(pngImage, "image.png", { type: "image/png" }),
    mask: await toFile(convertedMask, "mask.png", { type: "image/png" }),
    prompt,
    n: Math.min(n, 10),
  } as Parameters<typeof client.images.edit>[0]);

  const images = response.data ?? [];
  return images.map((item) => {
    if (!item.b64_json) {
      throw new Error("OpenAI did not return image data.");
    }
    return Buffer.from(item.b64_json, "base64");
  });
}
