"use client";

import { useRef } from "react";

export type BatchImage = {
  id: string;
  file: File;
  previewUrl: string;
  maskDataUrl: string | null;
};

type BatchPanelProps = {
  images: BatchImage[];
  activeIndex: number;
  onImagesAdd: (newImages: BatchImage[]) => void;
  onImageSelect: (index: number) => void;
  onClearAll: () => void;
  onRemoveImage: (index: number) => void;
  isRunning: boolean;
  uploadProgress: { uploaded: number; total: number } | null;
  batchStatus: { status: string; completedImages: number; failedImages: number; totalImages: number } | null;
  onRun: () => void;
  onAutoRun?: () => void;
  isAutoRunning?: boolean;
  onNew: () => void;
  singleJobActive: boolean;
  automaskMode?: "manual" | "auto";
  maskUrls?: (string | null)[];
  onMaskLightbox?: (url: string, caption: string) => void;
};

export default function BatchPanel({
  images,
  activeIndex,
  onImagesAdd,
  onImageSelect,
  onClearAll,
  onRemoveImage,
  isRunning,
  uploadProgress,
  batchStatus,
  onRun,
  onAutoRun,
  isAutoRunning = false,
  onNew,
  singleJobActive,
  automaskMode,
  maskUrls,
  onMaskLightbox,
}: BatchPanelProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const isAutoMask = automaskMode === "auto";
  const maskedCount = images.filter((img) => img.maskDataUrl !== null).length;
  const allMasked = images.length > 0 && (isAutoMask || maskedCount === images.length);

  const nextUnmaskedIndex = isAutoMask ? -1 : images.findIndex((img) => img.maskDataUrl === null);

  function handleFileInput(files: FileList | null) {
    if (!files) return;
    const imageFiles = Array.from(files).filter((f) =>
      f.type.startsWith("image/"),
    );
    if (imageFiles.length === 0) return;
    const newImages: BatchImage[] = imageFiles.map((file) => ({
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      file,
      previewUrl: URL.createObjectURL(file),
      maskDataUrl: null,
    }));
    onImagesAdd(newImages);
  }

  const runLabel = isRunning
    ? uploadProgress
      ? `Uploading… ${uploadProgress.uploaded}/${uploadProgress.total}`
      : "Processing…"
    : `▶ Run Batch (${images.length} images)`;

  return (
    <div className="batch-panel">
      {/* Upload controls */}
      <div className="batch-upload-zone">
        <div className="batch-upload-btns">
          <button
            className="btn btn-outline btn-sm"
            onClick={() => fileInputRef.current?.click()}
            disabled={isRunning}
          >
            +Files
          </button>
          <button
            className="btn btn-outline btn-sm"
            onClick={() => folderInputRef.current?.click()}
            disabled={isRunning}
          >
            +Folder
          </button>
          {images.length > 0 && (
            <button
              className="btn btn-outline btn-sm"
              style={{ color: "var(--red)", borderColor: "var(--red)" }}
              onClick={onClearAll}
              disabled={isRunning}
            >
              Clear All
            </button>
          )}
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          style={{ display: "none" }}
          onChange={(e) => handleFileInput(e.target.files)}
        />
        {/* folder input — webkitdirectory is non-standard but widely supported */}
        <input
          ref={folderInputRef}
          type="file"
          accept="image/*"
          // @ts-expect-error webkitdirectory is non-standard
          webkitdirectory=""
          style={{ display: "none" }}
          onChange={(e) => handleFileInput(e.target.files)}
        />
      </div>

      {/* Summary + progress */}
      {images.length > 0 && (
        <div className="batch-summary">
          <div className="batch-summary-row">
            <span className="batch-summary-label">Images</span>
            <span className="batch-summary-val">{images.length}</span>
          </div>
          {isAutoMask ? (
            <p className="hint" style={{ color: "var(--accent-light)", marginBottom: 2 }}>
              SAM2 auto-mask active — no manual masking needed
            </p>
          ) : (
            <>
              <div className="batch-summary-row">
                <span className="batch-summary-label">Masked</span>
                <span
                  className="batch-summary-val"
                  style={{ color: allMasked ? "var(--green)" : "var(--orange)" }}
                >
                  {maskedCount} / {images.length}
                </span>
              </div>
              {/* Masking progress bar */}
              <div className="batch-progress-bar-track">
                <div
                  className="batch-progress-bar-fill"
                  style={{ width: `${images.length > 0 ? (maskedCount / images.length) * 100 : 0}%` }}
                />
              </div>
              {/* Jump to next unmasked */}
              {nextUnmaskedIndex !== -1 && (
                <button
                  className="batch-jump-btn"
                  onClick={() => onImageSelect(nextUnmaskedIndex)}
                >
                  → Jump to next unmasked ({nextUnmaskedIndex + 1})
                </button>
              )}
            </>
          )}

          {/* Navigation */}
          <div className="batch-nav-row">
            <button
              className="btn btn-outline btn-sm"
              disabled={activeIndex <= 0}
              onClick={() => onImageSelect(activeIndex - 1)}
            >
              ← Prev
            </button>
            <span className="batch-nav-counter">
              {activeIndex + 1} / {images.length}
            </span>
            <button
              className="btn btn-outline btn-sm"
              disabled={activeIndex >= images.length - 1}
              onClick={() => onImageSelect(activeIndex + 1)}
            >
              Next →
            </button>
          </div>
        </div>
      )}

      {/* Batch run status */}
      {batchStatus && batchStatus.status !== "pending" && (
        <div className="batch-run-status">
          <div className="batch-summary-row">
            <span className="batch-summary-label">Status</span>
            <span
              className="batch-summary-val"
              style={{
                color:
                  batchStatus.status === "completed"
                    ? "var(--green)"
                    : batchStatus.status === "failed"
                    ? "var(--red)"
                    : "var(--orange)",
              }}
            >
              {batchStatus.status}
            </span>
          </div>
          <div className="batch-summary-row">
            <span className="batch-summary-label">Done</span>
            <span className="batch-summary-val">
              {batchStatus.completedImages} / {batchStatus.totalImages}
            </span>
          </div>
          {batchStatus.failedImages > 0 && (
            <div className="batch-summary-row">
              <span className="batch-summary-label">Failed</span>
              <span className="batch-summary-val" style={{ color: "var(--red)" }}>
                {batchStatus.failedImages}
              </span>
            </div>
          )}
          <div className="batch-progress-bar-track">
            <div
              className="batch-progress-bar-fill"
              style={{
                width: `${batchStatus.totalImages > 0
                  ? ((batchStatus.completedImages + batchStatus.failedImages) / batchStatus.totalImages) * 100
                  : 0}%`,
              }}
            />
          </div>
        </div>
      )}

      {/* Thumbnail grid */}
      {images.length > 0 && (
        <div className="batch-thumb-grid">
          {images.map((img, idx) => (
            <div
              key={img.id}
              className={`batch-thumb${idx === activeIndex ? " active" : ""}${img.maskDataUrl ? " masked" : ""}`}
              onClick={() => onImageSelect(idx)}
              title={img.file.name}
            >
              <img src={img.previewUrl} alt={img.file.name} />
              {maskUrls?.[idx] && (
                <img
                  src={maskUrls[idx]!}
                  alt="SAM2 mask"
                  className="batch-mask-thumb"
                  onClick={(e) => {
                    e.stopPropagation();
                    onMaskLightbox?.(maskUrls[idx]!, `SAM2 mask · ${img.file.name}`);
                  }}
                  style={onMaskLightbox ? { cursor: "zoom-in" } : undefined}
                />
              )}
              <div className="batch-thumb-status">
                {maskUrls?.[idx] ? "✓" : img.maskDataUrl ? "✓" : isAutoMask ? "⟳" : "○"}
              </div>
              {!isRunning && (
                <button
                  className="batch-thumb-remove"
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemoveImage(idx);
                  }}
                  title="Remove"
                >
                  ✕
                </button>
              )}
              <div className="batch-thumb-index">{idx + 1}</div>
            </div>
          ))}
        </div>
      )}


      {/* Run button at bottom */}
      {images.length > 0 && (
        <div className="batch-run-footer">
          {!allMasked && !isAutoMask && !isRunning && batchStatus?.status !== "completed" && (
            <p className="hint" style={{ marginBottom: 6, textAlign: "center" }}>
              Mask all {images.length - maskedCount} remaining image{images.length - maskedCount !== 1 ? "s" : ""} to enable Run.
            </p>
          )}
          {singleJobActive && (
            <p className="hint" style={{ marginBottom: 6, textAlign: "center", color: "var(--orange)" }}>
              Single job in progress
            </p>
          )}
          <div style={{ display: "flex", gap: 8, flexDirection: "column" }}>
            {/* Auto Run button — no mask required, SAM2 will generate masks */}
            {onAutoRun && (
              <button
                className={`btn-run${(isAutoRunning || isRunning) ? " running" : ""}`}
                style={{ background: "linear-gradient(135deg, var(--accent-dim), var(--accent))", fontSize: "0.8rem" }}
                disabled={images.length === 0 || isRunning || singleJobActive || isAutoRunning}
                onClick={onAutoRun}
                title="Auto-extract ODD factors & SAM2 target, then run batch (SAM2 generates masks automatically)"
              >
                {isAutoRunning ? "⟳  Setting up…" : isRunning ? "⟳  Processing…" : "✦ Auto Run →"}
              </button>
            )}
            <div style={{ display: "flex", gap: 8 }}>
              <button
                className={`btn-run${isRunning ? " running" : ""}`}
                style={{ flex: 1, fontSize: "0.75rem" }}
                disabled={!allMasked || isRunning || singleJobActive || isAutoRunning}
                onClick={onRun}
              >
                {runLabel}
              </button>
              {batchStatus?.status === "completed" && (
                <button
                  className="btn btn-outline"
                  style={{ whiteSpace: "nowrap" }}
                  onClick={onNew}
                >
                  New
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
