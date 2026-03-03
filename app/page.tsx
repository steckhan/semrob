"use client";

import { useEffect, useMemo, useState } from "react";

import MaskCanvas from "./components/MaskCanvas";

type JobOutput = {
  workflowName: string;
  variationIndex: number;
  url: string;
};

type JobRecord = {
  id: string;
  status: string;
  comfyBaseUrl?: string;
  error?: string;
  outputs: JobOutput[];
};

type WorkflowMapping = {
  workflowName: string;
  targets: {
    imageNodeId: string;
    imageInputKey?: string;
    maskNodeId: string;
    maskInputKey?: string;
    paramsNodeId: string;
  };
};

type WorkflowResponse = {
  workflows: WorkflowMapping[];
};

type MappingIssue = {
  workflowName: string;
  field: "imageNodeId" | "maskNodeId" | "paramsNodeId";
  message: string;
};

const COMFYUI_LOCAL_STORAGE_KEY = "comfyBaseUrl";
const DEFAULT_COMFYUI_BASE_URL = "http://172.26.224.1:8188";

const DEFAULT_PARAMS = {
  seed: 42,
  steps: 4,
  cfgScale: 1,
  sampler: "euler_ancestral",
  variationCount: 4,
  useWorkflowDefaults: false,
  positivePrompt: "",
  colorMatchStrength: 0.4,
  inpaintMode: "inpaint" as "inpaint" | "outpaint",
};


export default function HomePage() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [maskDataUrl, setMaskDataUrl] = useState<string | null>(null);
  const [maskedImageDataUrl, setMaskedImageDataUrl] = useState<string | null>(null);
  const [job, setJob] = useState<JobRecord | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [workflowMappings, setWorkflowMappings] = useState<WorkflowMapping[]>([]);
  const [comfyBaseUrl, setComfyBaseUrl] = useState(DEFAULT_COMFYUI_BASE_URL);
  const [isTestingComfy, setIsTestingComfy] = useState(false);
  const [comfyTestMessage, setComfyTestMessage] = useState<string | null>(null);
  const [comfyTestError, setComfyTestError] = useState<string | null>(null);

  useEffect(() => {
    const stored = window.localStorage.getItem(COMFYUI_LOCAL_STORAGE_KEY);
    if (!stored) {
      return;
    }

    const trimmed = stored.trim();
    if (!trimmed) {
      return;
    }

    setComfyBaseUrl(trimmed);
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

  useEffect(() => {
    const fetchMappings = async () => {
      const response = await fetch("/api/workflows");
      if (!response.ok) {
        return;
      }
      const payload = (await response.json()) as WorkflowResponse;
      setWorkflowMappings(payload.workflows);
    };
    void fetchMappings();
  }, []);

  const groupedOutputs = useMemo(() => {
    if (!job?.outputs) {
      return {} as Record<string, JobOutput[]>;
    }
    return job.outputs.reduce((acc, output) => {
      acc[output.workflowName] = [...(acc[output.workflowName] ?? []), output];
      return acc;
    }, {} as Record<string, JobOutput[]>);
  }, [job]);

  const submitJob = async () => {
    if (!imageFile || !maskDataUrl) {
      return;
    }
    setIsSubmitting(true);
    setJob(null);

    const formData = new FormData();
    // Send the RGBA combined image (original + mask as alpha) as the primary image.
    // ComfyUI's LoadImage reads the alpha channel as the inpaint mask.
    if (maskedImageDataUrl) {
      const rgbaBlob = await fetch(maskedImageDataUrl).then((res) => res.blob());
      formData.append("image", rgbaBlob, "input.png");
    } else {
      formData.append("image", imageFile);
    }
    const maskBlob = await fetch(maskDataUrl).then((res) => res.blob());
    formData.append("mask", maskBlob, "mask.png");
    Object.entries(params).forEach(([key, value]) => {
      formData.append(key, String(value));
    });
    formData.append("comfyBaseUrl", comfyBaseUrl.trim());

    const response = await fetch("/api/jobs", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const payload = (await response.json()) as { error?: string };
      console.error(payload.error ?? "Failed to submit job.");
      setComfyTestError(payload.error ?? "Failed to submit job.");
      setIsSubmitting(false);
      return;
    }

    const payload = (await response.json()) as JobRecord;
    setJob(payload);
    setComfyTestError(null);
    setComfyTestMessage("Job submitted successfully.");
    window.localStorage.setItem(COMFYUI_LOCAL_STORAGE_KEY, comfyBaseUrl.trim());
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
      setComfyTestError(payload.error ?? "Failed to connect to ComfyUI.");
      setIsTestingComfy(false);
      return;
    }

    const resolvedUrl = payload.comfyBaseUrl ?? comfyBaseUrl.trim();
    setComfyBaseUrl(resolvedUrl);
    window.localStorage.setItem(COMFYUI_LOCAL_STORAGE_KEY, resolvedUrl);
    setComfyTestMessage(`Connected to ${resolvedUrl}`);
    setIsTestingComfy(false);
  };

  useEffect(() => {
    if (!job) {
      return;
    }
    if (job.status === "completed" || job.status === "failed") {
      return;
    }

    const timer = window.setInterval(async () => {
      const response = await fetch(`/api/jobs/${job.id}`);
      if (!response.ok) {
        return;
      }
      const payload = (await response.json()) as JobRecord;
      setJob(payload);
    }, 1500);

    return () => window.clearInterval(timer);
  }, [job]);

  const updateMapping = (
    workflowName: string,
    updates: Partial<WorkflowMapping["targets"]>,
  ) => {
    setWorkflowMappings((current) =>
      current.map((mapping) =>
        mapping.workflowName === workflowName
          ? {
              ...mapping,
              targets: {
                ...mapping.targets,
                ...updates,
              },
            }
          : mapping,
      ),
    );
  };

  const validateMappings = (mappings: WorkflowMapping[]) => {
    const errors: string[] = [];
    mappings.forEach((mapping) => {
      if (!mapping.workflowName.trim()) {
        errors.push("Workflow name is missing.");
        return;
      }
      if (!mapping.targets.imageNodeId.trim()) {
        errors.push(`${mapping.workflowName}: image node ID is required.`);
      }
      if (!mapping.targets.maskNodeId.trim()) {
        errors.push(`${mapping.workflowName}: mask node ID is required.`);
      }
      if (!mapping.targets.paramsNodeId.trim()) {
        errors.push(`${mapping.workflowName}: params node ID is required.`);
      }
    });
    return errors;
  };

  const mappingIssues = useMemo(() => {
    const issues: MappingIssue[] = [];
    workflowMappings.forEach((mapping) => {
      if (!mapping.targets.imageNodeId.trim()) {
        issues.push({
          workflowName: mapping.workflowName,
          field: "imageNodeId",
          message: `${mapping.workflowName}: image node ID is required.`,
        });
      }
      if (!mapping.targets.maskNodeId.trim()) {
        issues.push({
          workflowName: mapping.workflowName,
          field: "maskNodeId",
          message: `${mapping.workflowName}: mask node ID is required.`,
        });
      }
      if (!mapping.targets.paramsNodeId.trim()) {
        issues.push({
          workflowName: mapping.workflowName,
          field: "paramsNodeId",
          message: `${mapping.workflowName}: params node ID is required.`,
        });
      }
    });
    return issues;
  }, [workflowMappings]);

  const mappingErrors = useMemo(
    () => validateMappings(workflowMappings),
    [workflowMappings],
  );

  const hasFieldIssue = (workflowName: string, field: MappingIssue["field"]) =>
    mappingIssues.some(
      (issue) => issue.workflowName === workflowName && issue.field === field,
    );

  const saveMappings = async () => {
    if (mappingErrors.length > 0) {
      return;
    }

    const response = await fetch("/api/workflows", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workflows: workflowMappings }),
    });

    if (!response.ok) {
      const payload = (await response.json()) as { error?: string };
      console.error(payload.error ?? "Failed to save workflow mappings.");
    }
  };

  return (
    <div className="panel">
      <div>
        <h1>ComfyUI Inpaint Studio</h1>
        <p>Upload an image, paint your mask, and run all workflows in parallel.</p>
      </div>

      <div className="panel">
        <h2>ComfyUI Connection</h2>
        <div className="row">
          <div style={{ flex: 1 }}>
            <label htmlFor="comfy-url">ComfyUI Base URL</label>
            <input
              id="comfy-url"
              className="input"
              type="url"
              value={comfyBaseUrl}
              onChange={(event) => setComfyBaseUrl(event.target.value)}
              placeholder="http://127.0.0.1:8188"
            />
          </div>
          <div style={{ display: "flex", alignItems: "flex-end" }}>
            <button
              type="button"
              className="button"
              onClick={testComfyConnection}
              disabled={isTestingComfy}
            >
              {isTestingComfy ? "Testing..." : "Test Connection"}
            </button>
          </div>
        </div>
        {comfyTestMessage && <p className="small">{comfyTestMessage}</p>}
        {comfyTestError && <p className="small">{comfyTestError}</p>}
      </div>

      <div className="panel">
        <label htmlFor="image">Image Upload</label>
        <input
          id="image"
          className="input"
          type="file"
          accept="image/*"
          onChange={(event) => setImageFile(event.target.files?.[0] ?? null)}
        />
      </div>

      <MaskCanvas
        imageUrl={imagePreview}
        onMaskReady={setMaskDataUrl}
        onMaskedImageReady={setMaskedImageDataUrl}
      />

      <div className="panel">
        <h2>Workflow Node Mapping</h2>
        <p className="small">
          Match node IDs and input keys for each workflow. Defaults are used if
          left unchanged.
        </p>
        {workflowMappings.length === 0 ? (
          <p className="small">No workflows detected yet.</p>
        ) : (
          <div className="grid">
            {workflowMappings.map((mapping) => (
              <div key={mapping.workflowName} className="panel">
                <strong>{mapping.workflowName}</strong>
                <div className="row">
                  <div style={{ flex: 1 }}>
                    <label>Image Node ID</label>
                    <input
                      className={`input ${
                        hasFieldIssue(mapping.workflowName, "imageNodeId")
                          ? "input-error"
                          : ""
                      }`}
                      value={mapping.targets.imageNodeId}
                      onChange={(event) =>
                        updateMapping(mapping.workflowName, {
                          imageNodeId: event.target.value,
                        })
                      }
                    />
                  </div>
                  <div style={{ flex: 1 }}>
                    <label>Image Input Key</label>
                    <input
                      className="input"
                      value={mapping.targets.imageInputKey ?? ""}
                      onChange={(event) =>
                        updateMapping(mapping.workflowName, {
                          imageInputKey: event.target.value,
                        })
                      }
                    />
                  </div>
                </div>
                <div className="row">
                  <div style={{ flex: 1 }}>
                    <label>Mask Node ID</label>
                    <input
                      className={`input ${
                        hasFieldIssue(mapping.workflowName, "maskNodeId")
                          ? "input-error"
                          : ""
                      }`}
                      value={mapping.targets.maskNodeId}
                      onChange={(event) =>
                        updateMapping(mapping.workflowName, {
                          maskNodeId: event.target.value,
                        })
                      }
                    />
                  </div>
                  <div style={{ flex: 1 }}>
                    <label>Mask Input Key</label>
                    <input
                      className="input"
                      value={mapping.targets.maskInputKey ?? ""}
                      onChange={(event) =>
                        updateMapping(mapping.workflowName, {
                          maskInputKey: event.target.value,
                        })
                      }
                    />
                  </div>
                </div>
                <div>
                  <label>Params Node ID</label>
                  <input
                    className={`input ${
                      hasFieldIssue(mapping.workflowName, "paramsNodeId")
                        ? "input-error"
                        : ""
                    }`}
                    value={mapping.targets.paramsNodeId}
                    onChange={(event) =>
                      updateMapping(mapping.workflowName, {
                        paramsNodeId: event.target.value,
                      })
                    }
                  />
                </div>
              </div>
            ))}
          </div>
        )}
        {mappingErrors.length > 0 && (
          <div className="panel">
            <strong>Mapping issues</strong>
            <ul className="small">
              {mappingErrors.map((error) => (
                <li key={error}>{error}</li>
              ))}
            </ul>
          </div>
        )}
        <button
          type="button"
          className="button"
          onClick={saveMappings}
          disabled={workflowMappings.length === 0 || mappingErrors.length > 0}
        >
          Save Workflow Mapping
        </button>
      </div>

      <div className="panel">
        <h2>Parameters</h2>
        <label className="row" style={{ alignItems: "center" }}>
          <input
            type="checkbox"
            checked={params.useWorkflowDefaults}
            onChange={(event) =>
              setParams((prev) => ({
                ...prev,
                useWorkflowDefaults: event.target.checked,
              }))
            }
          />
          <span>Use workflow defaults for sampler parameters</span>
        </label>
        <div className="row" style={{ alignItems: "center", gap: "0.75rem" }}>
          <span>Mode:</span>
          <label className="row" style={{ alignItems: "center", gap: "0.25rem" }}>
            <input
              type="radio"
              name="inpaintMode"
              value="inpaint"
              checked={params.inpaintMode === "inpaint"}
              onChange={() =>
                setParams((prev) => ({ ...prev, inpaintMode: "inpaint" }))
              }
            />
            <span>Inpaint</span>
          </label>
          <label className="row" style={{ alignItems: "center", gap: "0.25rem" }}>
            <input
              type="radio"
              name="inpaintMode"
              value="outpaint"
              checked={params.inpaintMode === "outpaint"}
              onChange={() =>
                setParams((prev) => ({ ...prev, inpaintMode: "outpaint" }))
              }
            />
            <span>Outpaint</span>
          </label>
        </div>
        <div className="grid">
          <div style={{ gridColumn: "1 / -1" }}>
            <label>Prompt</label>
            <input
              className="input"
              type="text"
              value={params.positivePrompt}
              disabled={params.useWorkflowDefaults}
              onChange={(event) =>
                setParams((prev) => ({
                  ...prev,
                  positivePrompt: event.target.value,
                }))
              }
            />
          </div>
          <div>
            <label>Seed</label>
            <input
              className="input"
              type="number"
              value={params.seed}
              disabled={params.useWorkflowDefaults}
              onChange={(event) =>
                setParams((prev) => ({
                  ...prev,
                  seed: Number(event.target.value),
                }))
              }
            />
          </div>
          <div>
            <label>Steps</label>
            <input
              className="input"
              type="number"
              min={1}
              value={params.steps}
              disabled={params.useWorkflowDefaults}
              onChange={(event) =>
                setParams((prev) => ({
                  ...prev,
                  steps: Number(event.target.value),
                }))
              }
            />
          </div>
          <div>
            <label>CFG</label>
            <input
              className="input"
              type="number"
              step={0.1}
              min={0}
              value={params.cfgScale}
              disabled={params.useWorkflowDefaults}
              onChange={(event) =>
                setParams((prev) => ({
                  ...prev,
                  cfgScale: Number(event.target.value),
                }))
              }
            />
          </div>
          <div>
            <label>Sampler</label>
            <select
              value={params.sampler}
              disabled={params.useWorkflowDefaults}
              onChange={(event) =>
                setParams((prev) => ({
                  ...prev,
                  sampler: event.target.value,
                }))
              }
            >
              <option value="euler_ancestral">euler_ancestral</option>
              <option value="euler">euler</option>
              <option value="dpmpp_2m">dpmpp_2m</option>
              <option value="dpmpp_sde">dpmpp_sde</option>
            </select>
          </div>
          <div>
            <label>Color Match Strength</label>
            <input
              className="input"
              type="number"
              step={0.05}
              min={0}
              max={1}
              value={params.colorMatchStrength}
              disabled={params.useWorkflowDefaults}
              onChange={(event) =>
                setParams((prev) => ({
                  ...prev,
                  colorMatchStrength: Number(event.target.value),
                }))
              }
            />
          </div>
          <div>
            <label>Variations</label>
            <input
              className="input"
              type="number"
              min={1}
              max={12}
              value={params.variationCount}
              onChange={(event) =>
                setParams((prev) => ({
                  ...prev,
                  variationCount: Number(event.target.value),
                }))
              }
            />
          </div>
        </div>
      </div>

      <button
        type="button"
        className="button"
        onClick={submitJob}
        disabled={!imageFile || !maskDataUrl || isSubmitting}
      >
        {isSubmitting ? "Submitting..." : "Run Inpainting"}
      </button>

      {job && (
        <div className="panel">
          <div className="row">
            <span className="status-pill">Status: {job.status}</span>
            <span className="small">Job ID: {job.id}</span>
          </div>
          {job.status === "failed" && job.error && (
            <p className="small">{job.error}</p>
          )}

          {job.status === "completed" && (
            <div className="gallery">
              {Object.entries(groupedOutputs).map(([workflowName, outputs]) => (
                <div key={workflowName}>
                  <h3>{workflowName}</h3>
                  <div className="gallery">
                    {outputs.map((output) => (
                      <img
                        key={`${workflowName}-${output.variationIndex}`}
                        src={output.url}
                        alt={`${workflowName} variation ${output.variationIndex}`}
                      />
                    ))}
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
