"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type { OddCatalog } from "@/lib/oddCatalog";
import type { SeedMode } from "@/lib/types";
import { buildPromptFromSelections } from "@/lib/oddCatalog";

import BatchPanel, { type BatchImage } from "./components/BatchPanel";
import ImageCompare from "./components/ImageCompare";
import MaskCanvas from "./components/MaskCanvas";
import OddDomainCard from "./components/OddDomainCard";
import YoloCompareModal from "./components/YoloCompareModal";
import OddFactorCard from "./components/OddFactorCard";

type JobOutput = {
  workflowName: string;
  variationIndex: number;
  url: string;
};

type YoloBox = {
  class: number;
  confidence: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
};

type YoloImageResult = {
  annotatedUrl: string;
  boxes: YoloBox[];
};

type YoloJobResults = {
  status: "running" | "completed" | "failed";
  model: string;
  confThreshold: number;
  original?: YoloImageResult;
  outputs?: Record<string, YoloImageResult>;
  error?: string;
};

type JobRecord = {
  id: string;
  status: string;
  createdAt?: string;
  startedAt?: string;
  completedAt?: string;
  inpaintMode?: "local" | "api";
  openaiModel?: string;
  comfyBaseUrl?: string;
  error?: string;
  outputs: JobOutput[];
  yoloResults?: YoloJobResults;
};

type BatchSubJobRecord = {
  imageIndex: number;
  originalName: string;
  jobId?: string;
  status: "pending" | "running" | "completed" | "failed";
  error?: string;
};

type BatchStatusRecord = {
  id: string;
  status: string;
  startedAt?: string;
  completedAt?: string;
  totalImages: number;
  completedImages: number;
  failedImages: number;
  subJobs: BatchSubJobRecord[];
};

function dataUrlToBlob(dataUrl: string): Blob {
  const [header, data] = dataUrl.split(",");
  const mime = header.match(/:(.*?);/)?.[1] ?? "image/png";
  const binary = atob(data);
  const arr = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) arr[i] = binary.charCodeAt(i);
  return new Blob([arr], { type: mime });
}

function formatDuration(startedAt?: string, completedAt?: string): string | null {
  if (!startedAt || !completedAt) return null;
  const ms = new Date(completedAt).getTime() - new Date(startedAt).getTime();
  if (ms < 0) return null;
  const totalSec = ms / 1000;
  if (totalSec < 60) return `${totalSec.toFixed(1)} s`;
  const m = Math.floor(totalSec / 60);
  const s = Math.round(totalSec % 60);
  return `${m}m ${s}s`;
}

const COMFYUI_LOCAL_STORAGE_KEY = "comfyBaseUrl";
const INPAINT_MODE_KEY = "inpaintMode";
const OPENAI_API_KEY_KEY = "openaiApiKey";
const OPENAI_MODEL_KEY = "openaiModel";
const ODD_DOMAIN_KEY = "oddDomain";
const ODD_CATALOG_CACHE_KEY = "oddCatalogCache";
const DEFAULT_COMFYUI_BASE_URL = "http://localhost:8188";
const CHUNK_SIZE = 20; // images per upload chunk

const OPENAI_MODELS = [
  { value: "gpt-image-1", label: "gpt-image-1" },
  { value: "gpt-image-1.5", label: "gpt-image-1.5" },
] as const;
type OpenAIModelValue = (typeof OPENAI_MODELS)[number]["value"];

const DEFAULT_PARAMS = {
  seed: 42,
  seedMode: "random" as SeedMode,
  steps: 4,
  cfgScale: 1,
  sampler: "euler_ancestral",
  scheduler: "normal",
  denoise: 1,
  maskStrength: 1,
  variationCount: 1,
  useWorkflowDefaults: false,
  positivePrompt: "",
  negativePrompt: "",
  automaskMode: "manual" as "manual" | "auto",
  sam2Prompt: "hand",
};

function JobResultSection({
  job,
  imagePreviewUrl,
  showDuration = true,
  onLightbox,
  onCompare,
  onYoloCompare,
}: {
  job: JobRecord;
  imagePreviewUrl: string | null;
  showDuration?: boolean;
  onLightbox: (url: string, caption: string) => void;
  onCompare: (url: string) => void;
  onYoloCompare: (orig: YoloImageResult, inpainted: YoloImageResult, label: string) => void;
}) {
  const grouped = job.outputs?.reduce(
    (acc, output) => {
      acc[output.workflowName] = [...(acc[output.workflowName] ?? []), output];
      return acc;
    },
    {} as Record<string, JobOutput[]>,
  ) ?? {};

  return (
    <>
      <div className="card">
        <div className="card-header">Job Status</div>
        <div className="card-body">
          <div className="metric-row">
            <span className="metric-label">Status</span>
            <span className={`metric-value ${job.status === "completed" ? "positive" : job.status === "failed" ? "negative" : "warning"}`}>{job.status}</span>
          </div>
          <div className="metric-row">
            <span className="metric-label">Backend</span>
            <span className="metric-value neutral">{job.inpaintMode === "api" ? (job.openaiModel ?? "OpenAI API") : "ComfyUI"}</span>
          </div>
          {showDuration && formatDuration(job.startedAt, job.completedAt) && (
            <div className="metric-row">
              <span className="metric-label">Duration</span>
              <span className="metric-value neutral">{formatDuration(job.startedAt, job.completedAt)}</span>
            </div>
          )}
          <div className="metric-row">
            <span className="metric-label">Job ID</span>
            <span className="metric-value neutral" style={{ fontSize: "0.62rem", fontWeight: 400 }}>{job.id.slice(0, 8)}…</span>
          </div>
          {job.status === "failed" && job.error && (
            <p className="hint" style={{ color: "var(--red)", marginTop: 4 }}>{job.error}</p>
          )}
        </div>
      </div>

      {job.status === "completed" && Object.entries(grouped).map(([workflowName, outputs]) => (
        <div key={workflowName} className="card">
          <div className="card-header">
            <span>{workflowName}</span>
            {showDuration && formatDuration(job.startedAt, job.completedAt) && (
              <span style={{ color: "var(--text-muted)", fontWeight: 500 }}>⏱ {formatDuration(job.startedAt, job.completedAt)}</span>
            )}
          </div>
          <div className="card-body">
            <div className="gallery">
              {outputs.map((output) => (
                <div
                  key={`${workflowName}-${output.variationIndex}`}
                  className="gallery-item"
                  onClick={() => onLightbox(output.url, `${workflowName} · variation ${output.variationIndex + 1}`)}
                >
                  <img src={output.url} alt={`${workflowName} variation ${output.variationIndex}`} />
                  {imagePreviewUrl && (
                    <button className="gallery-compare-btn" onClick={(e) => { e.stopPropagation(); onCompare(output.url); }}>
                      ⇄ Compare
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      ))}

      {job.yoloResults && (
        <div className="card">
          <div className="card-header">
            YOLO Detection
            {job.yoloResults.status === "running" && <span style={{ color: "var(--orange)", fontWeight: 500 }}>⟳ Analyzing…</span>}
          </div>
          <div className="card-body">
            {job.yoloResults.status === "failed" && (
              <p className="hint" style={{ color: "var(--red)" }}>{job.yoloResults.error ?? "YOLO detection failed."}</p>
            )}
            {job.yoloResults.status === "completed" && job.yoloResults.original && (
              <>
                <div>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
                    <span style={{ fontSize: "0.67rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.04em", color: "var(--text-dim)" }}>Original</span>
                    <span className={`detection-badge${job.yoloResults.original.boxes.length === 0 ? " clear" : ""}`}>
                      {job.yoloResults.original.boxes.length} detection{job.yoloResults.original.boxes.length !== 1 ? "s" : ""}
                    </span>
                  </div>
                  {job.yoloResults.original.annotatedUrl && (
                    <div className="gallery-item" onClick={() => onLightbox(job.yoloResults!.original!.annotatedUrl, "Original · YOLO annotated")}>
                      <img src={job.yoloResults.original.annotatedUrl} alt="Original YOLO annotated" />
                    </div>
                  )}
                </div>
                {job.yoloResults.outputs && Object.entries(job.yoloResults.outputs).map(([key, result]) => {
                  const count = result.boxes.length;
                  return (
                    <div key={key} style={{ marginTop: 8 }}>
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
                        <span style={{ fontSize: "0.62rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.04em", color: "var(--text-dim)" }}>{key}</span>
                        <span className={`detection-badge${count === 0 ? " clear" : ""}`}>{count} detection{count !== 1 ? "s" : ""}</span>
                      </div>
                      {result.annotatedUrl && (
                        <div className="gallery-item" onClick={() => onLightbox(result.annotatedUrl, `${key} · YOLO annotated`)}>
                          <img src={result.annotatedUrl} alt={`${key} YOLO annotated`} />
                          <button className="gallery-compare-btn" onClick={(e) => { e.stopPropagation(); onYoloCompare(job.yoloResults!.original!, result, key); }}>
                            ⇄ Compare
                          </button>
                        </div>
                      )}
                    </div>
                  );
                })}
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
}

function BatchResultCard({
  imageName,
  job,
  imagePreviewUrl,
  onLightbox,
  onYoloCompare,
}: {
  imageName: string;
  job: JobRecord;
  imagePreviewUrl: string | null;
  onLightbox: (url: string, caption: string) => void;
  onYoloCompare: (orig: YoloImageResult, inpainted: YoloImageResult, label: string) => void;
}) {
  const [showYoloImages, setShowYoloImages] = useState(false);

  const firstOutput = job.outputs?.[0];
  const yolo = job.yoloResults;
  const yoloDone = yolo?.status === "completed";
  const yoloRunning = yolo?.status === "running";

  return (
    <div className="card" style={{ marginBottom: 8 }}>
      {/* Header */}
      <div className="card-header" style={{ gap: 6, flexWrap: "wrap" }}>
        <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontSize: "0.67rem" }} title={imageName}>
          {imageName}
        </span>
        <span className={`metric-value ${job.status === "completed" ? "positive" : job.status === "failed" ? "negative" : "warning"}`} style={{ fontSize: "0.67rem" }}>
          {job.status}
        </span>
        {formatDuration(job.startedAt, job.completedAt) && (
          <span style={{ color: "var(--text-muted)", fontSize: "0.67rem" }}>
            {formatDuration(job.startedAt, job.completedAt)}
          </span>
        )}
      </div>

      {job.status === "failed" && (
        <div className="card-body" style={{ paddingTop: 6 }}>
          <p className="hint" style={{ color: "var(--red)" }}>{job.error ?? "Failed"}</p>
        </div>
      )}

      {job.status === "completed" && (
        <div className="card-body" style={{ paddingTop: 8 }}>
          {/* Thumbnail + YOLO summary row */}
          <div style={{ display: "grid", gridTemplateColumns: "72px 1fr", gap: 8, alignItems: "start" }}>
            {/* Inpainted thumbnail */}
            {firstOutput ? (
              <div
                className="gallery-item"
                style={{ width: 72, height: 56, cursor: "pointer" }}
                onClick={() => onLightbox(firstOutput.url, `${firstOutput.workflowName} · ${imageName}`)}
              >
                <img src={firstOutput.url} alt={imageName} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
              </div>
            ) : (
              <div style={{ width: 72, height: 56, background: "var(--surface2)", borderRadius: 4 }} />
            )}

            {/* YOLO detection summary */}
            <div>
              {yoloRunning && (
                <div style={{ fontSize: "0.67rem", color: "var(--orange)" }}>⟳ Analyzing…</div>
              )}
              {yoloDone && yolo?.original && (
                <>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
                    <span style={{ fontSize: "0.62rem", color: "var(--text-dim)" }}>Original</span>
                    <span className={`detection-badge${yolo.original.boxes.length === 0 ? " clear" : ""}`} style={{ fontSize: "0.6rem" }}>
                      {yolo.original.boxes.length} det
                    </span>
                  </div>
                  {yolo.outputs && Object.entries(yolo.outputs).map(([key, result]) => (
                    <div key={key} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
                      <span style={{ fontSize: "0.62rem", color: "var(--text-dim)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 80 }} title={key}>{key}</span>
                      <span className={`detection-badge${result.boxes.length === 0 ? " clear" : ""}`} style={{ fontSize: "0.6rem" }}>
                        {result.boxes.length} det
                      </span>
                    </div>
                  ))}
                  <button
                    className="btn btn-outline btn-sm"
                    style={{ marginTop: 4, fontSize: "0.6rem", padding: "2px 6px" }}
                    onClick={() => setShowYoloImages((v) => !v)}
                  >
                    {showYoloImages ? "▲ Hide" : "▼ YOLO images"}
                  </button>
                </>
              )}
              {yolo?.status === "failed" && (
                <span style={{ fontSize: "0.62rem", color: "var(--red)" }}>YOLO failed</span>
              )}
            </div>
          </div>

          {/* Expanded YOLO images */}
          {showYoloImages && yoloDone && yolo?.original && (
            <div style={{ marginTop: 8, borderTop: "1px solid var(--border)", paddingTop: 8 }}>
              {yolo.original.annotatedUrl && (
                <div style={{ marginBottom: 6 }}>
                  <div style={{ fontSize: "0.6rem", color: "var(--text-dim)", marginBottom: 3 }}>Original</div>
                  <div className="gallery-item" onClick={() => onLightbox(yolo!.original!.annotatedUrl, `Original · ${imageName}`)}>
                    <img src={yolo.original.annotatedUrl} alt="Original YOLO" />
                  </div>
                </div>
              )}
              {yolo.outputs && Object.entries(yolo.outputs).map(([key, result]) => result.annotatedUrl && (
                <div key={key} style={{ marginBottom: 6 }}>
                  <div style={{ fontSize: "0.6rem", color: "var(--text-dim)", marginBottom: 3 }}>{key}</div>
                  <div className="gallery-item" onClick={() => onLightbox(result.annotatedUrl, `${key} · ${imageName}`)}>
                    <img src={result.annotatedUrl} alt={key} />
                    <button className="gallery-compare-btn" onClick={(e) => { e.stopPropagation(); onYoloCompare(yolo!.original!, result, key); }}>
                      ⇄ Compare
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function HomePage() {
  // ── App mode ──────────────────────────────────────────────────────────────
  const [appMode, setAppMode] = useState<"single" | "batch">("single");

  // ── Single mode state ─────────────────────────────────────────────────────
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [maskDataUrl, setMaskDataUrl] = useState<string | null>(null);
  const [job, setJob] = useState<JobRecord | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // ── Batch mode state ──────────────────────────────────────────────────────
  const [batchImages, setBatchImages] = useState<BatchImage[]>([]);
  const [batchActiveIndex, setBatchActiveIndex] = useState(0);
  const [batchJobId, setBatchJobId] = useState<string | null>(null);
  const [batchStatus, setBatchStatus] = useState<BatchStatusRecord | null>(null);
  const [isBatchRunning, setIsBatchRunning] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<{ uploaded: number; total: number } | null>(null);
  const [batchSelectedJob, setBatchSelectedJob] = useState<JobRecord | null>(null);
  const [batchAllJobs, setBatchAllJobs] = useState<Record<number, JobRecord>>({});
  // Track whether we need to reset the canvas when switching batch images
  const prevBatchIndexRef = useRef<number>(-1);

  // ── Shared settings ───────────────────────────────────────────────────────
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [comfyBaseUrl, setComfyBaseUrl] = useState(DEFAULT_COMFYUI_BASE_URL);
  const [isTestingComfy, setIsTestingComfy] = useState(false);
  const [comfyTestMessage, setComfyTestMessage] = useState<string | null>(null);
  const [comfyTestError, setComfyTestError] = useState<string | null>(null);
  const [inpaintMode, setInpaintMode] = useState<"local" | "api">("local");
  const [openaiApiKey, setOpenaiApiKey] = useState("");
  const [openaiModel, setOpenaiModel] = useState<OpenAIModelValue>("gpt-image-1");

  // ── Modals ────────────────────────────────────────────────────────────────
  const [compareUrl, setCompareUrl] = useState<string | null>(null);
  const [yoloCompare, setYoloCompare] = useState<{
    originalResult: YoloImageResult;
    inpaintedResult: YoloImageResult;
    inpaintedLabel: string;
  } | null>(null);
  const [lightboxUrl, setLightboxUrl] = useState<string | null>(null);
  const [lightboxCaption, setLightboxCaption] = useState<string>("");

  // ── ODD ───────────────────────────────────────────────────────────────────
  const [oddDomain, setOddDomain] = useState("");
  const [oddCatalog, setOddCatalog] = useState<OddCatalog | null>(null);
  const [selectedFactorIds, setSelectedFactorIds] = useState<Set<string>>(new Set());
  const [isGeneratingOdd, setIsGeneratingOdd] = useState(false);
  const [isCustomPrompt, setIsCustomPrompt] = useState(false);
  const [oddError, setOddError] = useState<string | null>(null);

  // ── Lightbox Escape key ───────────────────────────────────────────────────
  useEffect(() => {
    if (!lightboxUrl) return;
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") setLightboxUrl(null); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [lightboxUrl]);

  // ── Load persisted settings ───────────────────────────────────────────────
  useEffect(() => {
    const stored = window.localStorage.getItem(COMFYUI_LOCAL_STORAGE_KEY);
    if (stored?.trim()) setComfyBaseUrl(stored.trim());

    const storedMode = window.localStorage.getItem(INPAINT_MODE_KEY);
    if (storedMode === "api" || storedMode === "local") setInpaintMode(storedMode);

    const storedKey = window.localStorage.getItem(OPENAI_API_KEY_KEY);
    if (storedKey) setOpenaiApiKey(storedKey);

    const storedModel = window.localStorage.getItem(OPENAI_MODEL_KEY);
    if (storedModel === "gpt-image-1" || storedModel === "gpt-image-1.5") {
      setOpenaiModel(storedModel);
    }

    const storedDomain = window.localStorage.getItem(ODD_DOMAIN_KEY);
    if (storedDomain) {
      setOddDomain(storedDomain);
      try {
        const cache = JSON.parse(
          window.localStorage.getItem(ODD_CATALOG_CACHE_KEY) ?? "{}",
        ) as Record<string, OddCatalog>;
        if (cache[storedDomain]) setOddCatalog(cache[storedDomain]);
      } catch { /* ignore corrupt cache */ }
    }
  }, []);

  useEffect(() => {
    if (!imageFile) { setImagePreview(null); return; }
    const url = URL.createObjectURL(imageFile);
    setImagePreview(url);
    return () => URL.revokeObjectURL(url);
  }, [imageFile]);

  // ── ODD generation ────────────────────────────────────────────────────────
  const generateOddCatalog = useCallback(async () => {
    const domain = oddDomain.trim();
    if (!domain) return;
    try {
      const cache = JSON.parse(
        window.localStorage.getItem(ODD_CATALOG_CACHE_KEY) ?? "{}",
      ) as Record<string, OddCatalog>;
      if (cache[domain]) {
        setOddCatalog(cache[domain]);
        setSelectedFactorIds(new Set());
        setIsCustomPrompt(false);
        setOddError(null);
        return;
      }
    } catch { /* ignore */ }

    setIsGeneratingOdd(true);
    setOddError(null);
    try {
      const response = await fetch("/api/odd/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ domain, apiKey: openaiApiKey.trim() || undefined }),
      });
      if (!response.ok) {
        const payload = (await response.json()) as { error?: string };
        setOddError(payload.error ?? "Failed to generate factors.");
        return;
      }
      const catalog = (await response.json()) as OddCatalog;
      setOddCatalog(catalog);
      setSelectedFactorIds(new Set());
      setIsCustomPrompt(false);
      try {
        const cache = JSON.parse(window.localStorage.getItem(ODD_CATALOG_CACHE_KEY) ?? "{}") as Record<string, OddCatalog>;
        cache[domain] = catalog;
        window.localStorage.setItem(ODD_CATALOG_CACHE_KEY, JSON.stringify(cache));
      } catch { /* ignore */ }
      window.localStorage.setItem(ODD_DOMAIN_KEY, domain);
    } catch (error) {
      setOddError(error instanceof Error ? error.message : "Network error.");
    } finally {
      setIsGeneratingOdd(false);
    }
  }, [oddDomain, openaiApiKey]);

  useEffect(() => {
    if (isCustomPrompt || !oddCatalog) return;
    const prompt = buildPromptFromSelections(oddCatalog, selectedFactorIds);
    setParams((p) => ({ ...p, positivePrompt: prompt }));
  }, [selectedFactorIds, oddCatalog, isCustomPrompt]);

  // The job to display in the right sidebar (single mode only)
  const displayJob = appMode === "single" ? job : null;

  // ── Single job submit ─────────────────────────────────────────────────────
  const submitJob = async () => {
    if (!imageFile || !maskDataUrl) return;
    setIsSubmitting(true);
    setJob(null);
    setComfyTestError(null);
    setComfyTestMessage(null);

    try {
      const formData = new FormData();
      formData.append("image", imageFile);
      const maskBlob = dataUrlToBlob(maskDataUrl);
      formData.append("mask", maskBlob, "mask.png");
      Object.entries(params).forEach(([key, value]) => formData.append(key, String(value)));
      formData.append("inpaintMode", inpaintMode);
      if (inpaintMode === "api") {
        formData.append("openaiApiKey", openaiApiKey.trim());
        formData.append("openaiModel", openaiModel);
      } else {
        formData.append("comfyBaseUrl", comfyBaseUrl.trim());
      }

      const response = await fetch("/api/jobs", { method: "POST", body: formData });
      if (!response.ok) {
        const payload = (await response.json()) as { error?: string };
        setComfyTestError(payload.error ?? "Failed to submit job.");
        return;
      }

      const payload = (await response.json()) as JobRecord;
      setJob(payload);
      window.localStorage.setItem(INPAINT_MODE_KEY, inpaintMode);
      if (inpaintMode === "local") {
        window.localStorage.setItem(COMFYUI_LOCAL_STORAGE_KEY, comfyBaseUrl.trim());
      } else {
        window.localStorage.setItem(OPENAI_API_KEY_KEY, openaiApiKey.trim());
        window.localStorage.setItem(OPENAI_MODEL_KEY, openaiModel);
      }
    } catch (err) {
      setComfyTestError(err instanceof Error ? err.message : "Network error submitting job.");
    } finally {
      setIsSubmitting(false);
    }
  };

  // ── Batch job submit ──────────────────────────────────────────────────────
  const submitBatch = async () => {
    if (batchImages.length === 0) return;
    setIsBatchRunning(true);
    setUploadProgress({ uploaded: 0, total: batchImages.length });

    try {
      // 1. Create batch record
      const batchFormData = new FormData();
      batchFormData.append("inpaintMode", inpaintMode);
      batchFormData.append("comfyBaseUrl", comfyBaseUrl.trim());
      batchFormData.append("openaiModel", openaiModel);
      Object.entries(params).forEach(([key, value]) => batchFormData.append(key, String(value)));

      const createRes = await fetch("/api/batch", { method: "POST", body: batchFormData });
      if (!createRes.ok) {
        const err = (await createRes.json()) as { error?: string };
        setComfyTestError(err.error ?? "Failed to create batch.");
        setIsBatchRunning(false);
        return;
      }
      const { batchId } = (await createRes.json()) as { batchId: string };
      setBatchJobId(batchId);

      // 2. Upload images in chunks
      let uploaded = 0;
      for (let chunkStart = 0; chunkStart < batchImages.length; chunkStart += CHUNK_SIZE) {
        const chunk = batchImages.slice(chunkStart, chunkStart + CHUNK_SIZE);
        const uploadForm = new FormData();
        uploadForm.append("startIndex", String(chunkStart));

        for (const img of chunk) {
          uploadForm.append("images[]", img.file);
          uploadForm.append("masks[]", img.maskDataUrl ?? "");
          uploadForm.append("names[]", img.file.name);
        }

        const upRes = await fetch(`/api/batch/${batchId}/upload`, {
          method: "POST",
          body: uploadForm,
        });
        if (!upRes.ok) {
          const err = (await upRes.json()) as { error?: string };
          setComfyTestError(err.error ?? "Upload failed.");
          setIsBatchRunning(false);
          return;
        }
        uploaded += chunk.length;
        setUploadProgress({ uploaded, total: batchImages.length });
      }

      // 3. Start processing
      setUploadProgress(null);
      const runRes = await fetch(`/api/batch/${batchId}/run`, { method: "POST" });
      if (!runRes.ok) {
        const err = (await runRes.json()) as { error?: string };
        setComfyTestError(err.error ?? "Failed to start batch.");
        setIsBatchRunning(false);
        return;
      }
    } catch (err) {
      setComfyTestError(err instanceof Error ? err.message : "Batch submission failed.");
      setIsBatchRunning(false);
    }
  };

  const testComfyConnection = async () => {
    setIsTestingComfy(true);
    setComfyTestMessage(null);
    setComfyTestError(null);
    try {
      const response = await fetch("/api/comfyui/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comfyBaseUrl: comfyBaseUrl.trim() }),
      });
      const payload = (await response.json()) as { ok?: boolean; comfyBaseUrl?: string; error?: string };
      if (!response.ok || !payload.ok) {
        setComfyTestError(payload.error ?? "Failed to connect.");
        return;
      }
      const resolvedUrl = payload.comfyBaseUrl ?? comfyBaseUrl.trim();
      setComfyBaseUrl(resolvedUrl);
      window.localStorage.setItem(COMFYUI_LOCAL_STORAGE_KEY, resolvedUrl);
      setComfyTestMessage(`Connected to ${resolvedUrl}`);
    } catch (err) {
      setComfyTestError(err instanceof Error ? err.message : "Network error.");
    } finally {
      setIsTestingComfy(false);
    }
  };

  // ── Poll single job ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!job) return;
    const yoloDone = job.yoloResults?.status === "completed" || job.yoloResults?.status === "failed";
    if ((job.status === "completed" && yoloDone) || job.status === "failed") return;
    const timer = window.setInterval(async () => {
      try {
        const response = await fetch(`/api/jobs/${job.id}`);
        if (!response.ok) return;
        const payload = (await response.json()) as JobRecord;
        setJob(payload);
      } catch { /* network hiccup — retry next tick */ }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job]);

  // ── Poll batch status ─────────────────────────────────────────────────────
  useEffect(() => {
    if (!batchJobId) return;
    if (batchStatus?.status === "completed" || batchStatus?.status === "failed") {
      setIsBatchRunning(false);
      return;
    }
    const timer = window.setInterval(async () => {
      try {
        const res = await fetch(`/api/batch/${batchJobId}`);
        if (!res.ok) return;
        const payload = (await res.json()) as BatchStatusRecord;
        setBatchStatus(payload);
        if (payload.status === "completed" || payload.status === "failed") {
          setIsBatchRunning(false);
        }
      } catch { /* network hiccup — retry next tick */ }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [batchJobId, batchStatus?.status]);

  // ── Fetch ALL batch sub-job details when batch completes ─────────────────
  useEffect(() => {
    if (!batchStatus || batchStatus.status !== "completed") return;
    const subJobsWithId = batchStatus.subJobs.filter((s) => s.jobId);
    if (subJobsWithId.length === 0) return;
    Promise.all(
      subJobsWithId.map(async (s) => {
        try {
          const res = await fetch(`/api/jobs/${s.jobId!}`);
          if (!res.ok) return null;
          return { index: s.imageIndex, job: (await res.json()) as JobRecord };
        } catch { return null; }
      }),
    ).then((results) => {
      const map: Record<number, JobRecord> = {};
      for (const r of results) {
        if (r) map[r.index] = r.job;
      }
      setBatchAllJobs(map);
    });
  }, [batchStatus?.status]);

  // ── Poll batchAllJobs entries that still have YOLO running ───────────────
  useEffect(() => {
    const runningEntries = Object.entries(batchAllJobs).filter(
      ([, j]) => j.yoloResults?.status === "running",
    );
    if (runningEntries.length === 0) return;
    const timer = window.setInterval(async () => {
      try {
        const updates = await Promise.all(
          runningEntries.map(async ([idx, j]) => {
            try {
              const res = await fetch(`/api/jobs/${j.id}`);
              if (!res.ok) return null;
              return { idx: Number(idx), job: (await res.json()) as JobRecord };
            } catch { return null; }
          }),
        );
        setBatchAllJobs((prev) => {
          const next = { ...prev };
          for (const u of updates) if (u) next[u.idx] = u.job;
          return next;
        });
      } catch { /* network hiccup — retry next tick */ }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [batchAllJobs]);

  // ── Fetch selected batch sub-job details ────────────────────────────────
  useEffect(() => {
    if (appMode !== "batch" || !batchStatus?.subJobs) {
      setBatchSelectedJob(null);
      return;
    }
    const subJob = batchStatus.subJobs[batchActiveIndex];
    if (!subJob?.jobId) {
      setBatchSelectedJob(null);
      return;
    }
    let cancelled = false;
    const fetchJob = async () => {
      try {
        const res = await fetch(`/api/jobs/${subJob.jobId}`);
        if (!res.ok || cancelled) return;
        const data = (await res.json()) as JobRecord;
        if (!cancelled) setBatchSelectedJob(data);
      } catch { /* ignore */ }
    };
    fetchJob();
    // Re-poll while YOLO is running
    const timer = window.setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${subJob.jobId}`);
        if (!res.ok || cancelled) return;
        const data = (await res.json()) as JobRecord;
        if (!cancelled) {
          setBatchSelectedJob(data);
          if (data.yoloResults?.status !== "running") {
            window.clearInterval(timer);
          }
        }
      } catch { /* network hiccup — retry next tick */ }
    }, 2000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [appMode, batchActiveIndex, batchStatus?.subJobs]);

  const handleImageChange = (file: File | undefined) => {
    setImageFile(file ?? null);
    setMaskDataUrl(null);
  };

  // ── Batch image/mask management ───────────────────────────────────────────
  const handleBatchMaskReady = useCallback(
    (maskDataUrlValue: string) => {
      setBatchImages((prev) => {
        const updated = [...prev];
        if (updated[batchActiveIndex]) {
          updated[batchActiveIndex] = { ...updated[batchActiveIndex], maskDataUrl: maskDataUrlValue };
        }
        return updated;
      });
    },
    [batchActiveIndex],
  );

  const handleBatchImagesAdd = useCallback((newImages: BatchImage[]) => {
    setBatchImages((prev) => [...prev, ...newImages]);
  }, []);

  const handleBatchRemoveImage = useCallback(
    (idx: number) => {
      setBatchImages((prev) => {
        const updated = prev.filter((_, i) => i !== idx);
        return updated;
      });
      setBatchActiveIndex((prev) => Math.max(0, prev > idx ? prev - 1 : prev));
    },
    [],
  );

  const handleBatchClearAll = useCallback(() => {
    setBatchImages([]);
    setBatchActiveIndex(0);
    setBatchJobId(null);
    setBatchStatus(null);
    setIsBatchRunning(false);
    setUploadProgress(null);
    setBatchAllJobs({});
  }, []);

  // Detect active batch image change to reset canvas key
  const batchCanvasKey = `batch-canvas-${batchActiveIndex}-${batchImages[batchActiveIndex]?.id ?? "empty"}`;

  // Hide floating toolbar while a job is actively running
  const toolbarHidden = appMode === "single"
    ? (isSubmitting || job?.status === "running")
    : (isBatchRunning || batchStatus?.status === "running" || batchStatus?.status === "completed");

  // Steps: Upload → Mask → Run → Results (single mode only)
  const stepDone = [!!imageFile, !!maskDataUrl, !!job, job?.status === "completed"];

  // ── Job active guards ─────────────────────────────────────────────────────
  const singleJobActive = isSubmitting || (!!job && job.status !== "completed" && job.status !== "failed");
  const batchJobActive = isBatchRunning;

  const singleRunLabel = isSubmitting ? "⟳  Running…" : "▶  Run Inpainting";

  // The image to show in the canvas area
  const canvasImageUrl = appMode === "batch"
    ? (batchImages[batchActiveIndex]?.previewUrl ?? null)
    : imagePreview;

  const canvasMaskReady = appMode === "batch" ? handleBatchMaskReady : setMaskDataUrl;

  return (
    <div className="app">
      {/* ── Lightbox ── */}
      {lightboxUrl && (
        <div className="lightbox-overlay" onClick={() => setLightboxUrl(null)}>
          <img className="lightbox-img" src={lightboxUrl} alt="Full size preview" onClick={(e) => e.stopPropagation()} />
          <button className="lightbox-close" onClick={() => setLightboxUrl(null)}>✕</button>
          {lightboxCaption && <div className="lightbox-caption">{lightboxCaption}</div>}
        </div>
      )}

      {/* ── Compare modals ── */}
      {compareUrl && imagePreview && (
        <ImageCompare originalUrl={imagePreview} inpaintedUrl={compareUrl} onClose={() => setCompareUrl(null)} />
      )}
      {yoloCompare && (
        <YoloCompareModal
          originalResult={yoloCompare.originalResult}
          inpaintedResult={yoloCompare.inpaintedResult}
          inpaintedLabel={yoloCompare.inpaintedLabel}
          onClose={() => setYoloCompare(null)}
        />
      )}

      {/* ── Top bar ── */}
      <header className="topbar">
        <div className="topbar-logo">
          <div className="topbar-icon">IS</div>
          <span>Inpaint Studio</span>
          <span style={{ fontWeight: 400, color: "var(--text-muted)", fontSize: "0.8rem" }}>
            ComfyUI &amp; OpenAI
          </span>
        </div>
        <div className="topbar-meta">
          <span className={`badge ${inpaintMode === "local" ? "badge-local" : "badge-api"}`}>
            {inpaintMode === "local" ? "Local · ComfyUI" : `Cloud · ${openaiModel}`}
          </span>
          {job && appMode === "single" && (
            <span
              className="badge"
              style={{
                background: job.status === "completed" ? "rgba(34,197,94,0.15)" : job.status === "failed" ? "rgba(239,68,68,0.15)" : "rgba(245,158,11,0.15)",
                color: job.status === "completed" ? "var(--green)" : job.status === "failed" ? "var(--red)" : "var(--orange)",
              }}
            >
              {job.status}
            </span>
          )}
          {batchStatus && appMode === "batch" && (
            <span
              className="badge"
              style={{
                background: batchStatus.status === "completed" ? "rgba(34,197,94,0.15)" : batchStatus.status === "failed" ? "rgba(239,68,68,0.15)" : "rgba(245,158,11,0.15)",
                color: batchStatus.status === "completed" ? "var(--green)" : batchStatus.status === "failed" ? "var(--red)" : "var(--orange)",
              }}
            >
              Batch: {batchStatus.status}
            </span>
          )}
        </div>
      </header>

      {/* ── Steps bar (single mode) / mode tab switcher ── */}
      <nav className="steps-bar">
        {appMode === "single" ? (
          <>
            {(["Upload", "Mask", "Run", "Results"] as const).map((label, i) => {
              const done = stepDone[i];
              const active = !done && (i === 0 || stepDone[i - 1]);
              return (
                <div key={label} style={{ display: "contents" }}>
                  {i > 0 && <div className="step-connector" />}
                  <div className={`step-item${done ? " done" : active ? " active" : ""}`}>
                    <div className="step-num">{done ? "✓" : i + 1}</div>
                    {label}
                  </div>
                </div>
              );
            })}
            <div style={{ flex: 1 }} />
          </>
        ) : (
          <div style={{ flex: 1 }} />
        )}
        {/* Mode switcher tabs (always visible at right) */}
        <div className="app-mode-tabs">
          <button
            className={`app-mode-tab${appMode === "single" ? " active" : ""}`}
            onClick={() => setAppMode("single")}
            disabled={singleJobActive || batchJobActive}
            title={batchJobActive ? "Batch job in progress" : singleJobActive ? "Single job in progress" : undefined}
          >
            Single
          </button>
          <button
            className={`app-mode-tab${appMode === "batch" ? " active" : ""}`}
            onClick={() => setAppMode("batch")}
            disabled={singleJobActive || batchJobActive}
            title={singleJobActive ? "Single job in progress" : batchJobActive ? "Batch job in progress" : undefined}
          >
            Batch
          </button>
        </div>
      </nav>

      {/* ── Main layout ── */}
      <div className="main">
        {/* ════ Left sidebar ════ */}
        <div className="sidebar-left">
          <div className="sidebar-scroll">

            {/* ── Connection (always on top, always visible) ── */}
            {inpaintMode === "local" ? (
              <div className="card">
                <div className="card-header">ComfyUI Connection</div>
                <div className="card-body">
                  <div>
                    <label>Base URL</label>
                    <div className="row" style={{ flexWrap: "nowrap" }}>
                      <input
                        className="input"
                        type="url"
                        value={comfyBaseUrl}
                        onChange={(e) => setComfyBaseUrl(e.target.value)}
                        placeholder="http://127.0.0.1:8188"
                      />
                      <button className="btn btn-outline btn-sm" onClick={testComfyConnection} disabled={isTestingComfy}>
                        {isTestingComfy ? "…" : "Test"}
                      </button>
                    </div>
                  </div>
                  {comfyTestMessage && <p className="hint" style={{ color: "var(--green)" }}>{comfyTestMessage}</p>}
                </div>
              </div>
            ) : (
              <div className="card">
                <div className="card-header">OpenAI Settings</div>
                <div className="card-body">
                  <div>
                    <label>Model</label>
                    <div className="seg-control">
                      {OPENAI_MODELS.map((m) => (
                        <button key={m.value} className={`seg-btn${openaiModel === m.value ? " active" : ""}`} onClick={() => setOpenaiModel(m.value)}>
                          {m.label}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div>
                    <label>API Key</label>
                    <input className="input" type="password" value={openaiApiKey} onChange={(e) => setOpenaiApiKey(e.target.value)} placeholder="sk-… (or set OPENAI_API_KEY env var)" />
                  </div>
                  <p className="hint">Leave blank to use server-side env var.</p>
                </div>
              </div>
            )}

            {/* ── Mode toggle ── */}
            <div className="card">
              <div className="card-header">Mode</div>
              <div className="card-body">
                <div className="seg-control">
                  <button className={`seg-btn${inpaintMode === "local" ? " active" : ""}`} onClick={() => setInpaintMode("local")}>
                    Local (ComfyUI)
                  </button>
                  <button className={`seg-btn${inpaintMode === "api" ? " active" : ""}`} onClick={() => setInpaintMode("api")}>
                    OpenAI API
                  </button>
                </div>
              </div>
            </div>

            {/* ── Parameters ── */}
            <div className="card">
              <div className="card-header">Generation Parameters</div>
              <div className="card-body">
                {inpaintMode === "local" && (
                  <label className="row" style={{ textTransform: "none", letterSpacing: 0, fontSize: "0.72rem", cursor: "pointer" }}>
                    <input
                      type="checkbox"
                      checked={params.useWorkflowDefaults}
                      onChange={(e) => setParams((p) => ({ ...p, useWorkflowDefaults: e.target.checked }))}
                    />
                    Use workflow defaults
                  </label>
                )}

                <div>
                  <label>Prompt</label>
                  <input
                    className="input"
                    type="text"
                    value={params.positivePrompt}
                    disabled={inpaintMode === "local" && params.useWorkflowDefaults}
                    onChange={(e) => { setParams((p) => ({ ...p, positivePrompt: e.target.value })); setIsCustomPrompt(true); }}
                  />
                  {isCustomPrompt && oddCatalog && (
                    <button className="odd-clear-btn" style={{ marginTop: 3, fontSize: "0.6rem" }} onClick={() => setIsCustomPrompt(false)}>
                      Use ODD factors
                    </button>
                  )}
                </div>

                {appMode === "single" && (
                  <div className="slider-row">
                    <label className="slider-label">Variations</label>
                    <input type="range" min={1} max={inpaintMode === "api" ? 10 : 12} value={params.variationCount}
                      onChange={(e) => setParams((p) => ({ ...p, variationCount: Number(e.target.value) }))} />
                    <span className="slider-val">{params.variationCount}</span>
                  </div>
                )}

                {inpaintMode === "local" && (
                  <>
                    <div>
                      <label>Negative Prompt</label>
                      <input className="input" type="text" value={params.negativePrompt} disabled={params.useWorkflowDefaults}
                        onChange={(e) => setParams((p) => ({ ...p, negativePrompt: e.target.value }))} />
                    </div>
                    <div>
                      <label>Mask Mode</label>
                      <select className="input" value={params.automaskMode}
                        onChange={(e) => setParams((p) => ({ ...p, automaskMode: e.target.value as "manual" | "auto" }))}>
                        <option value="manual">Manual</option>
                        <option value="auto">Auto (SAM2)</option>
                      </select>
                    </div>
                    {params.automaskMode === "auto" && (
                      <div>
                        <label>SAM2 Segment Target</label>
                        <input
                          className="input"
                          type="text"
                          placeholder="e.g. hand, person, car"
                          value={params.sam2Prompt}
                          onChange={(e) => setParams((p) => ({ ...p, sam2Prompt: e.target.value }))}
                        />
                      </div>
                    )}
                    <div className="slider-row">
                      <label className="slider-label">Steps</label>
                      <input type="range" min={1} max={100} value={params.steps} disabled={params.useWorkflowDefaults}
                        onChange={(e) => setParams((p) => ({ ...p, steps: Number(e.target.value) }))} />
                      <span className="slider-val">{params.steps}</span>
                    </div>
                    <div className="slider-row">
                      <label className="slider-label">Mask Str.</label>
                      <input type="range" min={0} max={1} step={0.05} value={params.maskStrength}
                        onChange={(e) => setParams((p) => ({ ...p, maskStrength: Number(e.target.value) }))} />
                      <span className="slider-val">{params.maskStrength.toFixed(2)}</span>
                    </div>
                    <div className="params-grid">
                      <div>
                        <label>Sampler</label>
                        <select value={params.sampler} disabled={params.useWorkflowDefaults} onChange={(e) => setParams((p) => ({ ...p, sampler: e.target.value }))}>
                          <option value="euler_ancestral">euler_ancestral</option>
                          <option value="euler">euler</option>
                          <option value="euler_a">euler_a</option>
                          <option value="dpmpp_2m">dpmpp_2m</option>
                          <option value="dpmpp_sde">dpmpp_sde</option>
                        </select>
                      </div>
                      <div>
                        <label>Scheduler</label>
                        <select value={params.scheduler} disabled={params.useWorkflowDefaults} onChange={(e) => setParams((p) => ({ ...p, scheduler: e.target.value }))}>
                          <option value="normal">normal</option>
                          <option value="karras">karras</option>
                          <option value="simple">simple</option>
                        </select>
                      </div>
                    </div>
                    <div>
                      <label>Seed</label>
                      <div style={{ display: "flex", gap: 6 }}>
                        <select className="input" value={params.seedMode} disabled={params.useWorkflowDefaults}
                          onChange={(e) => setParams((p) => ({ ...p, seedMode: e.target.value as "random" | "increment" | "fixed" }))}>
                          <option value="random">random</option>
                          <option value="increment">increment</option>
                          <option value="fixed">fixed</option>
                        </select>
                        {params.seedMode !== "random" && (
                          <input className="input" type="number" value={params.seed} disabled={params.useWorkflowDefaults}
                            onChange={(e) => setParams((p) => ({ ...p, seed: Number(e.target.value) }))} />
                        )}
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* ── ODD Analysis — only in API mode ── */}
            {inpaintMode === "api" && (
              <details className="accordion">
                <summary>
                  <span>ODD Analysis (Advanced)</span>
                  <span style={{ fontSize: "0.6rem", opacity: 0.5 }}>▾</span>
                </summary>
                <div className="accordion-body">
                  <OddDomainCard domain={oddDomain} onDomainChange={setOddDomain} onGenerate={generateOddCatalog} isGenerating={isGeneratingOdd} />
                  {oddError && <p className="hint" style={{ color: "var(--red)" }}>{oddError}</p>}
                  <OddFactorCard catalog={oddCatalog} selectedFactorIds={selectedFactorIds} onSelectionChange={setSelectedFactorIds} />
                </div>
              </details>
            )}

            {/* ── Batch panel (shown when in batch mode) ── */}
            {appMode === "batch" && (
              <BatchPanel
                images={batchImages}
                activeIndex={batchActiveIndex}
                onImagesAdd={handleBatchImagesAdd}
                onImageSelect={setBatchActiveIndex}
                onClearAll={handleBatchClearAll}
                onRemoveImage={handleBatchRemoveImage}
                isRunning={isBatchRunning}
                uploadProgress={uploadProgress}
                batchStatus={batchStatus}
                onRun={submitBatch}
                onNew={handleBatchClearAll}
                singleJobActive={singleJobActive}
              />
            )}
          </div>

          {/* ── Sticky run button (single mode only) ── */}
          {appMode === "single" && (
            <div className="sidebar-footer">
              {comfyTestError && <p className="hint" style={{ color: "var(--red)", marginBottom: 8 }}>{comfyTestError}</p>}
              {batchJobActive && <p className="hint" style={{ color: "var(--orange)", marginBottom: 8 }}>Batch job in progress</p>}
              <div style={{ display: "flex", gap: 8 }}>
                <button
                  className={`btn-run${isSubmitting ? " running" : ""}`}
                  style={{ flex: 1 }}
                  onClick={submitJob}
                  disabled={!imageFile || !maskDataUrl || isSubmitting || batchJobActive}
                >
                  {singleRunLabel}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* ════ Center canvas ════ */}
        <div className="canvas-center">
          {!canvasImageUrl ? (
            <label className="upload-zone" htmlFor={appMode === "single" ? "image-upload" : undefined}
              onClick={appMode === "batch" ? undefined : undefined}>
              <span style={{ fontSize: "2.5rem", opacity: 0.2 }}>🖼</span>
              <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--text-dim)" }}>
                {appMode === "batch" ? "Select an image from the list to mask it" : "Click to upload an image"}
              </span>
              <span className="hint">{appMode === "single" ? "PNG · JPG · WebP" : "Use the Batch panel on the left to upload images"}</span>
              {appMode === "single" && (
                <input id="image-upload" type="file" accept="image/*" style={{ display: "none" }}
                  onChange={(e) => handleImageChange(e.target.files?.[0])} />
              )}
            </label>
          ) : (
            <>
              <div className="canvas-scroll">
                <MaskCanvas
                  key={appMode === "batch" ? batchCanvasKey : "single-canvas"}
                  imageUrl={canvasImageUrl}
                  onMaskReady={canvasMaskReady}
                  hideControls={toolbarHidden}
                />
              </div>
              <div className="canvas-bottom-bar">
                {appMode === "single" ? (
                  <>
                    <label htmlFor="image-upload-change" className="btn btn-outline btn-sm" style={{ cursor: "pointer" }}>
                      Change image
                    </label>
                    <input id="image-upload-change" type="file" accept="image/*" style={{ display: "none" }}
                      onChange={(e) => handleImageChange(e.target.files?.[0])} />
                    <span className="hint" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
                      {imageFile?.name}
                    </span>
                  </>
                ) : (
                  <>
                    <button
                      className="btn btn-outline btn-sm"
                      disabled={batchActiveIndex <= 0}
                      onClick={() => setBatchActiveIndex((i) => i - 1)}
                    >
                      ← Prev
                    </button>
                    <span className="hint" style={{ flex: 1, textAlign: "center" }}>
                      {batchImages[batchActiveIndex]?.file.name ?? ""}
                      {" "}·{" "}
                      <span style={{ color: batchImages[batchActiveIndex]?.maskDataUrl ? "var(--green)" : "var(--orange)" }}>
                        {batchImages[batchActiveIndex]?.maskDataUrl ? "Masked ✓" : "Not masked"}
                      </span>
                    </span>
                    <button
                      className="btn btn-outline btn-sm"
                      disabled={batchActiveIndex >= batchImages.length - 1}
                      onClick={() => setBatchActiveIndex((i) => i + 1)}
                    >
                      Next →
                    </button>
                  </>
                )}
              </div>
            </>
          )}
        </div>

        {/* ════ Right sidebar — results ════ */}
        <div className="sidebar-right">
          {appMode === "batch" && batchStatus && (
            <div className="card">
              <div className="card-header">Batch Progress</div>
              <div className="card-body">
                <div className="metric-row">
                  <span className="metric-label">Status</span>
                  <span className={`metric-value ${batchStatus.status === "completed" ? "positive" : batchStatus.status === "failed" ? "negative" : "warning"}`}>
                    {batchStatus.status}
                  </span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Completed</span>
                  <span className="metric-value neutral">{batchStatus.completedImages} / {batchStatus.totalImages}</span>
                </div>
                {batchStatus.failedImages > 0 && (
                  <div className="metric-row">
                    <span className="metric-label">Failed</span>
                    <span className="metric-value negative">{batchStatus.failedImages}</span>
                  </div>
                )}
                {formatDuration(batchStatus.startedAt, batchStatus.completedAt) && (
                  <div className="metric-row">
                    <span className="metric-label">Duration</span>
                    <span className="metric-value neutral">{formatDuration(batchStatus.startedAt, batchStatus.completedAt)}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── Single mode results ── */}
          {appMode === "single" && (!displayJob ? (
            <div className="empty-state">
              <span style={{ fontSize: "2rem", opacity: 0.15 }}>◷</span>
              <p className="hint" style={{ textAlign: "center" }}>Upload an image, paint your mask, and hit Run.</p>
            </div>
          ) : (
            <JobResultSection
              job={displayJob}
              imagePreviewUrl={imagePreview}
              onLightbox={(url, caption) => { setLightboxUrl(url); setLightboxCaption(caption); }}
              onCompare={(url) => setCompareUrl(url)}
              onYoloCompare={(orig, inpainted, label) => setYoloCompare({ originalResult: orig, inpaintedResult: inpainted, inpaintedLabel: label })}
            />
          ))}

          {/* ── Batch mode results ── */}
          {appMode === "batch" && (
            Object.keys(batchAllJobs).length > 0 ? (
              batchStatus!.subJobs
                .filter((s) => batchAllJobs[s.imageIndex])
                .map((s) => (
                  <BatchResultCard
                    key={s.imageIndex}
                    imageName={s.originalName}
                    job={batchAllJobs[s.imageIndex]}
                    imagePreviewUrl={batchImages[s.imageIndex]?.previewUrl ?? null}
                    onLightbox={(url, caption) => { setLightboxUrl(url); setLightboxCaption(caption); }}
                    onYoloCompare={(orig, inpainted, label) => setYoloCompare({ originalResult: orig, inpaintedResult: inpainted, inpaintedLabel: label })}
                  />
                ))
            ) : !batchStatus ? (
              <div className="empty-state">
                <span style={{ fontSize: "2rem", opacity: 0.15 }}>◷</span>
                <p className="hint" style={{ textAlign: "center" }}>Run a batch to see results here.</p>
              </div>
            ) : (
              <div className="empty-state">
                <span style={{ fontSize: "2rem", opacity: 0.15 }}>⟳</span>
                <p className="hint" style={{ textAlign: "center" }}>Processing… results will appear when complete.</p>
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
}
