"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import type { OddCatalog } from "@/lib/oddCatalog";
import { buildPromptFromSelections } from "@/lib/oddCatalog";

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

const OPENAI_MODELS = [
  { value: "gpt-image-1", label: "gpt-image-1" },
  { value: "gpt-image-1.5", label: "gpt-image-1.5" },
] as const;
type OpenAIModelValue = (typeof OPENAI_MODELS)[number]["value"];

const DEFAULT_PARAMS = {
  seed: 42,
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
};

export default function HomePage() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [maskDataUrl, setMaskDataUrl] = useState<string | null>(null);
  const [job, setJob] = useState<JobRecord | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [comfyBaseUrl, setComfyBaseUrl] = useState(DEFAULT_COMFYUI_BASE_URL);
  const [isTestingComfy, setIsTestingComfy] = useState(false);
  const [comfyTestMessage, setComfyTestMessage] = useState<string | null>(null);
  const [comfyTestError, setComfyTestError] = useState<string | null>(null);
  const [inpaintMode, setInpaintMode] = useState<"local" | "api">("local");
  const [openaiApiKey, setOpenaiApiKey] = useState("");
  const [openaiModel, setOpenaiModel] = useState<OpenAIModelValue>("gpt-image-1");
  const [compareUrl, setCompareUrl] = useState<string | null>(null);
  const [yoloCompare, setYoloCompare] = useState<{
    originalResult: YoloImageResult;
    inpaintedResult: YoloImageResult;
    inpaintedLabel: string;
  } | null>(null);
  const [oddDomain, setOddDomain] = useState("");
  const [oddCatalog, setOddCatalog] = useState<OddCatalog | null>(null);
  const [selectedFactorIds, setSelectedFactorIds] = useState<Set<string>>(new Set());
  const [isGeneratingOdd, setIsGeneratingOdd] = useState(false);
  const [isCustomPrompt, setIsCustomPrompt] = useState(false);
  const [oddError, setOddError] = useState<string | null>(null);

  // Load persisted settings
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
        if (cache[storedDomain]) {
          setOddCatalog(cache[storedDomain]);
        }
      } catch {
        // ignore corrupt cache
      }
    }
  }, []);


  useEffect(() => {
    if (!imageFile) {
      setImagePreview(null);
      return;
    }
    const url = URL.createObjectURL(imageFile);
    setImagePreview(url);
    return () => URL.revokeObjectURL(url);
  }, [imageFile]);


  // Generate ODD factors from domain description
  const generateOddCatalog = useCallback(async () => {
    const domain = oddDomain.trim();
    if (!domain) return;

    // Check localStorage cache first
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
    } catch {
      // ignore corrupt cache
    }

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
        setIsGeneratingOdd(false);
        return;
      }

      const catalog = (await response.json()) as OddCatalog;
      setOddCatalog(catalog);
      setSelectedFactorIds(new Set());
      setIsCustomPrompt(false);

      // Cache in localStorage
      try {
        const cache = JSON.parse(
          window.localStorage.getItem(ODD_CATALOG_CACHE_KEY) ?? "{}",
        ) as Record<string, OddCatalog>;
        cache[domain] = catalog;
        window.localStorage.setItem(ODD_CATALOG_CACHE_KEY, JSON.stringify(cache));
      } catch {
        // ignore storage errors
      }

      window.localStorage.setItem(ODD_DOMAIN_KEY, domain);
    } catch (error) {
      setOddError(error instanceof Error ? error.message : "Network error.");
    } finally {
      setIsGeneratingOdd(false);
    }
  }, [oddDomain, openaiApiKey]);

  // Auto-generate prompt from selected ODD factors
  useEffect(() => {
    if (isCustomPrompt || !oddCatalog) return;
    const prompt = buildPromptFromSelections(oddCatalog, selectedFactorIds);
    setParams((p) => ({ ...p, positivePrompt: prompt }));
  }, [selectedFactorIds, oddCatalog, isCustomPrompt]);

  const groupedOutputs = useMemo(() => {
    if (!job?.outputs) return {} as Record<string, JobOutput[]>;
    return job.outputs.reduce(
      (acc, output) => {
        acc[output.workflowName] = [...(acc[output.workflowName] ?? []), output];
        return acc;
      },
      {} as Record<string, JobOutput[]>,
    );
  }, [job]);

  const submitJob = async () => {
    if (!imageFile || !maskDataUrl) return;
    setIsSubmitting(true);
    setJob(null);
    setComfyTestError(null);
    setComfyTestMessage(null);

    const formData = new FormData();
    formData.append("image", imageFile);
    const maskBlob = await fetch(maskDataUrl).then((res) => res.blob());
    formData.append("mask", maskBlob, "mask.png");
    Object.entries(params).forEach(([key, value]) => {
      formData.append(key, String(value));
    });
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
      setIsSubmitting(false);
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
    setIsSubmitting(false);
  };

  const testComfyConnection = async () => {
    setIsTestingComfy(true);
    setComfyTestMessage(null);
    setComfyTestError(null);

    const response = await fetch("/api/comfyui/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comfyBaseUrl: comfyBaseUrl.trim() }),
    });

    const payload = (await response.json()) as {
      ok?: boolean;
      comfyBaseUrl?: string;
      error?: string;
    };

    if (!response.ok || !payload.ok) {
      setComfyTestError(payload.error ?? "Failed to connect.");
      setIsTestingComfy(false);
      return;
    }

    const resolvedUrl = payload.comfyBaseUrl ?? comfyBaseUrl.trim();
    setComfyBaseUrl(resolvedUrl);
    window.localStorage.setItem(COMFYUI_LOCAL_STORAGE_KEY, resolvedUrl);
    setComfyTestMessage(`Connected to ${resolvedUrl}`);
    setIsTestingComfy(false);
  };

  // Poll job status (including while YOLO runs after completion)
  useEffect(() => {
    if (!job) return;
    const yoloDone =
      job.yoloResults?.status === "completed" ||
      job.yoloResults?.status === "failed";
    if ((job.status === "completed" && yoloDone) || job.status === "failed") return;
    const timer = window.setInterval(async () => {
      const response = await fetch(`/api/jobs/${job.id}`);
      if (!response.ok) return;
      const payload = (await response.json()) as JobRecord;
      setJob(payload);
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job]);


  const handleImageChange = (file: File | undefined) => {
    setImageFile(file ?? null);
    setMaskDataUrl(null);
  };

  // Steps: Upload → Mask → Run → Results
  const stepDone = [!!imageFile, !!maskDataUrl, !!job, job?.status === "completed"];

  return (
    <div className="app">
      {/* ── Before/After compare modal ── */}
      {compareUrl && imagePreview && (
        <ImageCompare
          originalUrl={imagePreview}
          inpaintedUrl={compareUrl}
          onClose={() => setCompareUrl(null)}
        />
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
          <span
            style={{ fontWeight: 400, color: "var(--text-muted)", fontSize: "0.8rem" }}
          >
            ComfyUI &amp; OpenAI
          </span>
        </div>
        <div className="topbar-meta">
          <span
            className={`badge ${inpaintMode === "local" ? "badge-local" : "badge-api"}`}
          >
            {inpaintMode === "local" ? "Local · ComfyUI" : `Cloud · ${openaiModel}`}
          </span>
          {job && (
            <span
              className="badge"
              style={{
                background:
                  job.status === "completed"
                    ? "rgba(34,197,94,0.15)"
                    : job.status === "failed"
                      ? "rgba(239,68,68,0.15)"
                      : "rgba(245,158,11,0.15)",
                color:
                  job.status === "completed"
                    ? "var(--green)"
                    : job.status === "failed"
                      ? "var(--red)"
                      : "var(--orange)",
              }}
            >
              {job.status}
            </span>
          )}
        </div>
      </header>

      {/* ── Steps bar ── */}
      <nav className="steps-bar">
        {(["Upload", "Mask", "Run", "Results"] as const).map((label, i) => {
          const done = stepDone[i];
          const active = !done && (i === 0 || stepDone[i - 1]);
          return (
            <div key={label} style={{ display: "contents" }}>
              {i > 0 && <div className="step-connector" />}
              <div
                className={`step-item${done ? " done" : active ? " active" : ""}`}
              >
                <div className="step-num">{done ? "✓" : i + 1}</div>
                {label}
              </div>
            </div>
          );
        })}
      </nav>

      {/* ── Main 3-column layout ── */}
      <div className="main">
        {/* ════ Left sidebar — controls ════ */}
        <div className="sidebar-left">
          {/* Mode toggle */}
          <div className="card">
            <div className="card-header">Mode</div>
            <div className="card-body">
              <div className="seg-control">
                <button
                  className={`seg-btn${inpaintMode === "local" ? " active" : ""}`}
                  onClick={() => setInpaintMode("local")}
                >
                  Local (ComfyUI)
                </button>
                <button
                  className={`seg-btn${inpaintMode === "api" ? " active" : ""}`}
                  onClick={() => setInpaintMode("api")}
                >
                  OpenAI API
                </button>
              </div>
            </div>
          </div>

          {/* Connection */}
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
                    <button
                      className="btn btn-outline btn-sm"
                      onClick={testComfyConnection}
                      disabled={isTestingComfy}
                    >
                      {isTestingComfy ? "…" : "Test"}
                    </button>
                  </div>
                </div>
                {comfyTestMessage && (
                  <p className="hint" style={{ color: "var(--green)" }}>
                    {comfyTestMessage}
                  </p>
                )}
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
                      <button
                        key={m.value}
                        className={`seg-btn${openaiModel === m.value ? " active" : ""}`}
                        onClick={() => setOpenaiModel(m.value)}
                      >
                        {m.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label>API Key</label>
                  <input
                    className="input"
                    type="password"
                    value={openaiApiKey}
                    onChange={(e) => setOpenaiApiKey(e.target.value)}
                    placeholder="sk-… (or set OPENAI_API_KEY env var)"
                  />
                </div>
                <p className="hint">Leave blank to use server-side env var.</p>
              </div>
            </div>
          )}

          {/* ODD Domain */}
          <OddDomainCard
            domain={oddDomain}
            onDomainChange={setOddDomain}
            onGenerate={generateOddCatalog}
            isGenerating={isGeneratingOdd}
          />
          {oddError && (
            <p className="hint" style={{ color: "var(--red)" }}>
              {oddError}
            </p>
          )}

          {/* ODD Factors */}
          <OddFactorCard
            catalog={oddCatalog}
            selectedFactorIds={selectedFactorIds}
            onSelectionChange={setSelectedFactorIds}
          />

          {/* Parameters */}
          <div className="card">
            <div className="card-header">Parameters</div>
            <div className="card-body">
              {inpaintMode === "local" && (
                <label
                  className="row"
                  style={{
                    textTransform: "none",
                    letterSpacing: 0,
                    fontSize: "0.72rem",
                    cursor: "pointer",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={params.useWorkflowDefaults}
                    onChange={(e) =>
                      setParams((p) => ({
                        ...p,
                        useWorkflowDefaults: e.target.checked,
                      }))
                    }
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
                  onChange={(e) => {
                    setParams((p) => ({ ...p, positivePrompt: e.target.value }));
                    setIsCustomPrompt(true);
                  }}
                />
                {isCustomPrompt && oddCatalog && (
                  <button
                    className="odd-clear-btn"
                    style={{ marginTop: 3, fontSize: "0.6rem" }}
                    onClick={() => setIsCustomPrompt(false)}
                  >
                    Use ODD factors
                  </button>
                )}
              </div>

              <div className="slider-row">
                <label className="slider-label">Variations</label>
                <input
                  type="range"
                  min={1}
                  max={inpaintMode === "api" ? 10 : 12}
                  value={params.variationCount}
                  onChange={(e) =>
                    setParams((p) => ({
                      ...p,
                      variationCount: Number(e.target.value),
                    }))
                  }
                />
                <span className="slider-val">{params.variationCount}</span>
              </div>

              {inpaintMode === "local" && (
                <>
                  <div>
                    <label>Negative Prompt</label>
                    <input
                      className="input"
                      type="text"
                      value={params.negativePrompt}
                      disabled={params.useWorkflowDefaults}
                      onChange={(e) =>
                        setParams((p) => ({ ...p, negativePrompt: e.target.value }))
                      }
                    />
                  </div>

                  <div className="slider-row">
                    <label className="slider-label">Steps</label>
                    <input
                      type="range"
                      min={1}
                      max={100}
                      value={params.steps}
                      disabled={params.useWorkflowDefaults}
                      onChange={(e) =>
                        setParams((p) => ({ ...p, steps: Number(e.target.value) }))
                      }
                    />
                    <span className="slider-val">{params.steps}</span>
                  </div>

                  <div className="slider-row">
                    <label className="slider-label">Mask Str.</label>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={params.maskStrength}
                      onChange={(e) =>
                        setParams((p) => ({
                          ...p,
                          maskStrength: Number(e.target.value),
                        }))
                      }
                    />
                    <span className="slider-val">{params.maskStrength.toFixed(2)}</span>
                  </div>

                  <div className="params-grid">
                    <div>
                      <label>Sampler</label>
                      <select
                        value={params.sampler}
                        disabled={params.useWorkflowDefaults}
                        onChange={(e) =>
                          setParams((p) => ({ ...p, sampler: e.target.value }))
                        }
                      >
                        <option value="euler_ancestral">euler_ancestral</option>
                        <option value="euler">euler</option>
                        <option value="euler_a">euler_a</option>
                        <option value="dpmpp_2m">dpmpp_2m</option>
                        <option value="dpmpp_sde">dpmpp_sde</option>
                      </select>
                    </div>
                    <div>
                      <label>Scheduler</label>
                      <select
                        value={params.scheduler}
                        disabled={params.useWorkflowDefaults}
                        onChange={(e) =>
                          setParams((p) => ({ ...p, scheduler: e.target.value }))
                        }
                      >
                        <option value="normal">normal</option>
                        <option value="karras">karras</option>
                        <option value="simple">simple</option>
                      </select>
                    </div>
                  </div>

                  <div>
                    <label>Seed</label>
                    <input
                      className="input"
                      type="number"
                      value={params.seed}
                      disabled={params.useWorkflowDefaults}
                      onChange={(e) =>
                        setParams((p) => ({ ...p, seed: Number(e.target.value) }))
                      }
                    />
                  </div>
                </>
              )}
            </div>
          </div>


          {/* Run button */}
          <button
            className="btn btn-primary btn-full"
            onClick={submitJob}
            disabled={!imageFile || !maskDataUrl || isSubmitting}
          >
            {isSubmitting ? "⟳  Running…" : "▶  Run Inpainting"}
          </button>

          {comfyTestError && (
            <p className="hint" style={{ color: "var(--red)" }}>
              {comfyTestError}
            </p>
          )}
        </div>

        {/* ════ Center — canvas area ════ */}
        <div className="canvas-center">
          {!imageFile ? (
            <label className="upload-zone" htmlFor="image-upload">
              <span style={{ fontSize: "2.5rem", opacity: 0.2 }}>🖼</span>
              <span
                style={{
                  fontSize: "0.85rem",
                  fontWeight: 600,
                  color: "var(--text-dim)",
                }}
              >
                Click to upload an image
              </span>
              <span className="hint">PNG · JPG · WebP</span>
              <input
                id="image-upload"
                type="file"
                accept="image/*"
                style={{ display: "none" }}
                onChange={(e) => handleImageChange(e.target.files?.[0])}
              />
            </label>
          ) : (
            <>
              <div className="canvas-scroll">
                <MaskCanvas imageUrl={imagePreview} onMaskReady={setMaskDataUrl} />
              </div>
              <div className="canvas-bottom-bar">
                <label
                  htmlFor="image-upload-change"
                  className="btn btn-outline btn-sm"
                  style={{ cursor: "pointer" }}
                >
                  Change image
                </label>
                <input
                  id="image-upload-change"
                  type="file"
                  accept="image/*"
                  style={{ display: "none" }}
                  onChange={(e) => handleImageChange(e.target.files?.[0])}
                />
                <span
                  className="hint"
                  style={{
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    flex: 1,
                  }}
                >
                  {imageFile.name}
                </span>
              </div>
            </>
          )}
        </div>

        {/* ════ Right sidebar — results ════ */}
        <div className="sidebar-right">
          {!job ? (
            <div className="empty-state">
              <span style={{ fontSize: "2rem", opacity: 0.15 }}>◷</span>
              <p className="hint" style={{ textAlign: "center" }}>
                Upload an image, paint your mask, and hit Run.
              </p>
            </div>
          ) : (
            <>
              <div className="card">
                <div className="card-header">Job Status</div>
                <div className="card-body">
                  <div className="metric-row">
                    <span className="metric-label">Status</span>
                    <span
                      className={`metric-value ${
                        job.status === "completed"
                          ? "positive"
                          : job.status === "failed"
                            ? "negative"
                            : "warning"
                      }`}
                    >
                      {job.status}
                    </span>
                  </div>
                  <div className="metric-row">
                    <span className="metric-label">Backend</span>
                    <span className="metric-value neutral">
                      {job.inpaintMode === "api"
                        ? (job.openaiModel ?? "OpenAI API")
                        : "ComfyUI"}
                    </span>
                  </div>
                  {formatDuration(job.startedAt, job.completedAt) && (
                    <div className="metric-row">
                      <span className="metric-label">Duration</span>
                      <span className="metric-value neutral">
                        {formatDuration(job.startedAt, job.completedAt)}
                      </span>
                    </div>
                  )}
                  <div className="metric-row">
                    <span className="metric-label">Job ID</span>
                    <span
                      className="metric-value neutral"
                      style={{ fontSize: "0.62rem", fontWeight: 400 }}
                    >
                      {job.id.slice(0, 8)}…
                    </span>
                  </div>
                  {job.status === "failed" && job.error && (
                    <p className="hint" style={{ color: "var(--red)", marginTop: 4 }}>
                      {job.error}
                    </p>
                  )}
                </div>
              </div>

              {job.status === "completed" &&
                Object.entries(groupedOutputs).map(([workflowName, outputs]) => (
                  <div key={workflowName} className="card">
                    <div className="card-header">
                      <span>{workflowName}</span>
                      {formatDuration(job.startedAt, job.completedAt) && (
                        <span style={{ color: "var(--text-muted)", fontWeight: 500 }}>
                          ⏱ {formatDuration(job.startedAt, job.completedAt)}
                        </span>
                      )}
                    </div>
                    <div className="card-body">
                      <div className="gallery">
                        {outputs.map((output) => (
                          <div
                            key={`${workflowName}-${output.variationIndex}`}
                            className="gallery-item"
                          >
                            <img
                              src={output.url}
                              alt={`${workflowName} variation ${output.variationIndex}`}
                            />
                            {imagePreview && (
                              <button
                                className="gallery-compare-btn"
                                onClick={() => setCompareUrl(output.url)}
                              >
                                ⇄ Compare
                              </button>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}

              {/* YOLO Detection card */}
              {job.yoloResults && (
                <div className="card">
                  <div className="card-header">
                    YOLO Detection
                    {job.yoloResults.status === "running" && (
                      <span style={{ color: "var(--orange)", fontWeight: 500 }}>⟳ Analyzing…</span>
                    )}
                  </div>
                  <div className="card-body">
                    {job.yoloResults.status === "failed" && (
                      <p className="hint" style={{ color: "var(--red)" }}>
                        {job.yoloResults.error ?? "YOLO detection failed."}
                      </p>
                    )}
                    {job.yoloResults.status === "completed" && job.yoloResults.original && (
                      <>
                        {/* Original thumbnail */}
                        <div>
                          <div className="card-header" style={{ border: "none", padding: "0 0 6px 0", background: "none" }}>
                            <span>Original</span>
                            <span className={`metric-value ${job.yoloResults.original.boxes.length > 0 ? "negative" : "positive"}`} style={{ fontSize: "0.7rem" }}>
                              {job.yoloResults.original.boxes.length} detection{job.yoloResults.original.boxes.length !== 1 ? "s" : ""}
                            </span>
                          </div>
                          {job.yoloResults.original.annotatedUrl && (
                            <div className="gallery-item">
                              <img
                                src={job.yoloResults.original.annotatedUrl}
                                alt="Original YOLO annotated"
                              />
                            </div>
                          )}
                        </div>

                        {/* Output thumbnails — each with a Compare button */}
                        {job.yoloResults.outputs &&
                          Object.entries(job.yoloResults.outputs).map(([key, result]) => {
                            const count = result.boxes.length;
                            return (
                              <div key={key} style={{ marginTop: 8 }}>
                                <div className="card-header" style={{ border: "none", padding: "0 0 6px 0", background: "none" }}>
                                  <span style={{ fontSize: "0.62rem" }}>{key}</span>
                                  <span className={`metric-value ${count === 0 ? "positive" : "negative"}`} style={{ fontSize: "0.7rem" }}>
                                    {count} detection{count !== 1 ? "s" : ""}
                                  </span>
                                </div>
                                {result.annotatedUrl && (
                                  <div className="gallery-item">
                                    <img
                                      src={result.annotatedUrl}
                                      alt={`${key} YOLO annotated`}
                                    />
                                    <button
                                      className="gallery-compare-btn"
                                      onClick={() =>
                                        setYoloCompare({
                                          originalResult: job.yoloResults!.original!,
                                          inpaintedResult: result,
                                          inpaintedLabel: key,
                                        })
                                      }
                                    >
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
          )}
        </div>
      </div>
    </div>
  );
}
