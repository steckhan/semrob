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
 * Embeds the inpaint mask directly into the source image's alpha channel.
 *
 * gpt-image-1 uses the alpha channel of the *image* to identify the edit region:
 *   alpha = 0   (transparent) → model rewrites this pixel
 *   alpha = 255 (opaque)      → model keeps this pixel
 *
 * Our canvas mask convention: white (R > 128) = inpaint area, black = keep area.
 * So we set alpha=0 where the mask is white and alpha=255 where it is black.
 *
 * A separate `mask` parameter is NOT used — gpt-image-1 ignores it.
 */
async function embedMaskIntoImage(
  imageBuffer: Buffer,
  maskBuffer: Buffer
): Promise<Buffer> {
  const meta = await sharp(imageBuffer).metadata();
  const { width, height } = meta;
  if (!width || !height) throw new Error("Could not read image dimensions.");

  const { data: imgData } = await sharp(imageBuffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  // Resize mask to match image dimensions (in case they differ) and convert to greyscale
  const { data: maskData } = await sharp(maskBuffer)
    .resize(width, height, { fit: "fill" })
    .grayscale()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const rgba = Buffer.from(imgData);
  for (let i = 0; i < maskData.length; i++) {
    // White mask pixel (> 128) → inpaint area → transparent in image
    rgba[i * 4 + 3] = maskData[i] > 128 ? 0 : 255;
  }

  return sharp(rgba, {
    raw: { width, height, channels: 4 },
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
  // The mask is baked into the image's alpha channel — no separate mask file needed.
  const imageWithMask = await embedMaskIntoImage(imageBuffer, maskBuffer);

  const client = new OpenAI({ apiKey });

  const response = await client.images.edit({
    model,
    image: await toFile(imageWithMask, "image.png", { type: "image/png" }),
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
