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
 * Converts our canvas mask (white = inpaint area, black = keep area)
 * to the RGBA PNG format the images/edits endpoint expects:
 *   alpha = 0   (transparent) → model edits this area
 *   alpha = 255 (opaque)      → model keeps this area
 *
 * NOTE: As of mid-2025, gpt-image-1 has a confirmed model-level bug where it
 * ignores the mask and regenerates the entire image. This is the correct API
 * format per OpenAI docs; the limitation is in the model, not the code.
 * See: https://community.openai.com/t/image-editing-inpainting-with-a-mask-for-gpt-image-1-replaces-the-entire-image/1244275
 */
async function convertMaskForOpenAI(
  maskBuffer: Buffer,
  width: number,
  height: number
): Promise<Buffer> {
  const { data: maskData } = await sharp(maskBuffer)
    .resize(width, height, { fit: "fill" })
    .grayscale()
    .raw()
    .toBuffer({ resolveWithObject: true });

  // Build a fresh RGBA buffer: RGB=0, alpha driven by mask brightness
  const rgba = Buffer.alloc(width * height * 4);
  for (let i = 0; i < maskData.length; i++) {
    rgba[i * 4] = 0;
    rgba[i * 4 + 1] = 0;
    rgba[i * 4 + 2] = 0;
    rgba[i * 4 + 3] = maskData[i] > 128 ? 0 : 255; // white=inpaint→transparent, black=keep→opaque
  }

  return sharp(rgba, {
    raw: { width, height, channels: 4 },
  })
    .png()
    .toBuffer();
}

/**
 * Picks the closest supported output size for the images/edits endpoint.
 *
 * Supported presets and their aspect ratios:
 *   1024×1024  → 1.00  (square)
 *   1536×1024  → 1.50  (landscape)
 *   1024×1536  → 0.667 (portrait)
 *
 * Decision boundaries are the midpoints between adjacent ratios:
 *   > 1.25  → landscape
 *   < 0.833 → portrait
 *   else    → square
 */
function selectOutputSize(width: number, height: number): "1024x1024" | "1536x1024" | "1024x1536" {
  const ratio = width / height;
  if (ratio >= 1.25) return "1536x1024";
  if (ratio <= 0.833) return "1024x1536";
  return "1024x1024";
}

export async function runOpenAIInpainting({
  imageBuffer,
  maskBuffer,
  prompt,
  apiKey,
  n = 1,
  model = "gpt-image-1",
}: OpenAIInpaintInput): Promise<Buffer[]> {
  const meta = await sharp(imageBuffer).metadata();
  const { width, height } = meta;
  if (!width || !height) throw new Error("Could not read image dimensions.");

  const size = selectOutputSize(width, height);

  const [pngImage, convertedMask] = await Promise.all([
    sharp(imageBuffer).png().toBuffer(),
    convertMaskForOpenAI(maskBuffer, width, height),
  ]);

  const client = new OpenAI({ apiKey });

  const response = await client.images.edit({
    model,
    image: await toFile(pngImage, "image.png", { type: "image/png" }),
    mask: await toFile(convertedMask, "mask.png", { type: "image/png" }),
    prompt,
    n: Math.min(n, 10),
    size,
  } as Parameters<typeof client.images.edit>[0]);

  const images = response.data ?? [];
  return images.map((item) => {
    if (!item.b64_json) {
      throw new Error("OpenAI did not return image data.");
    }
    return Buffer.from(item.b64_json, "base64");
  });
}
