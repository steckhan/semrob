import fs from "fs/promises";
import path from "path";

import { NextResponse } from "next/server";
import sharp from "sharp";

import { DATA_ROOT } from "@/lib/constants";
import { readBatch, writeBatch } from "@/lib/batchRunner";
import type { BatchSubJob } from "@/lib/types";

const BATCH_DIR = path.join(DATA_ROOT, "batch");

/**
 * POST /api/batch/[batchId]/upload
 * Accepts a chunk of image+mask pairs as FormData.
 * FormData fields:
 *   images[]  — image Files
 *   masks[]   — mask data URLs (base64) matched by index
 *   names[]   — original filenames matched by index
 *   startIndex — the offset to use for naming files (for chunked uploads)
 */
export async function POST(
  request: Request,
  { params }: { params: Promise<{ batchId: string }> },
) {
  const { batchId } = await params;
  const batch = await readBatch(batchId);
  if (!batch) {
    return NextResponse.json({ error: "Batch not found" }, { status: 404 });
  }
  if (batch.status !== "pending" && batch.status !== "uploading") {
    return NextResponse.json(
      { error: "Batch is no longer accepting uploads" },
      { status: 400 },
    );
  }

  const formData = await request.formData();
  const startIndex = Number(formData.get("startIndex") ?? batch.subJobs.length);
  const imageFiles = formData.getAll("images[]") as File[];
  const maskDataUrls = formData.getAll("masks[]") as string[];
  const names = formData.getAll("names[]") as string[];
  const gtFiles = formData.getAll("gtFiles[]") as (File | string)[];

  if (imageFiles.length === 0) {
    return NextResponse.json({ error: "No images provided" }, { status: 400 });
  }

  const batchImagesDir = path.join(BATCH_DIR, batchId);
  await fs.mkdir(batchImagesDir, { recursive: true });

  const newSubJobs: BatchSubJob[] = [];

  for (let i = 0; i < imageFiles.length; i++) {
    const globalIndex = startIndex + i;
    const imageFile = imageFiles[i];
    const maskDataUrl = maskDataUrls[i] ?? "";
    const originalName = names[i] ?? imageFile.name;

    // Convert image to PNG via sharp for consistency
    const imageArrayBuffer = await imageFile.arrayBuffer();
    const imageBuffer = await sharp(Buffer.from(imageArrayBuffer)).png().toBuffer();

    // Decode mask data URL (base64 PNG)
    let maskBuffer: Buffer;
    if (maskDataUrl.startsWith("data:")) {
      const base64 = maskDataUrl.split(",")[1];
      maskBuffer = Buffer.from(base64, "base64");
    } else {
      // Empty mask — black image (no inpainting area)
      const meta = await sharp(imageBuffer).metadata();
      const w = meta.width ?? 512;
      const h = meta.height ?? 512;
      maskBuffer = await sharp(Buffer.alloc(w * h, 0), {
        raw: { width: w, height: h, channels: 1 },
      })
        .png()
        .toBuffer();
    }

    const imagePath = path.join(batchImagesDir, `image_${globalIndex}.png`);
    const maskPath = path.join(batchImagesDir, `mask_${globalIndex}.png`);
    await fs.writeFile(imagePath, imageBuffer);
    await fs.writeFile(maskPath, maskBuffer);

    // Save GT label file if provided and non-empty
    const gtEntry = gtFiles[i];
    if (gtEntry && typeof gtEntry !== "string" && gtEntry.size > 0) {
      const gtBuffer = Buffer.from(await gtEntry.arrayBuffer());
      const gtPath = path.join(batchImagesDir, `gt_${globalIndex}.txt`);
      await fs.writeFile(gtPath, gtBuffer);
    }

    newSubJobs.push({
      imageIndex: globalIndex,
      originalName,
      status: "pending",
    });
  }

  const updatedBatch = {
    ...batch,
    status: "uploading" as const,
    subJobs: [...batch.subJobs, ...newSubJobs],
    totalImages: batch.subJobs.length + newSubJobs.length,
  };

  await writeBatch(updatedBatch);
  return NextResponse.json({
    uploaded: newSubJobs.length,
    total: updatedBatch.totalImages,
  });
}
