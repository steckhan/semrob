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

  const DEFAULT_PARAMS = {
    seed: 42,
    steps: 28,
    cfgScale: 8,
    sampler: "euler",
    scheduler: "normal",
    denoise: 1,
    maskStrength: 1,
    variationCount: 4,
    useWorkflowDefaults: false,
    positivePrompt: "wristwatch, metal casing, worn look",
    negativePrompt: "",
  };


export default function HomePage() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [maskDataUrl, setMaskDataUrl] = useState<string | null>(null);
  const [job, setJob] = useState<JobRecord | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [workflowMappings, setWorkflowMappings] = useState<WorkflowMapping[]>([]);

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
    formData.append("image", imageFile);
    const maskBlob = await fetch(maskDataUrl).then((res) => res.blob());
    formData.append("mask", maskBlob, "mask.png");
    Object.entries(params).forEach(([key, value]) => {
      formData.append(key, String(value));
    });

    const response = await fetch("/api/jobs", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const payload = (await response.json()) as { error?: string };
      console.error(payload.error ?? "Failed to submit job.");
      setIsSubmitting(false);
      return;
    }

    const payload = (await response.json()) as JobRecord;
    setJob(payload);
    setIsSubmitting(false);
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
        <label htmlFor="image">Image Upload</label>
        <input
          id="image"
          className="input"
          type="file"
          accept="image/*"
          onChange={(event) => setImageFile(event.target.files?.[0] ?? null)}
        />
      </div>

      <MaskCanvas imageUrl={imagePreview} onMaskReady={setMaskDataUrl} />

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
        <div className="grid">
          <div style={{ gridColumn: "1 / -1" }}>
            <label>Positive Prompt</label>
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
          <div style={{ gridColumn: "1 / -1" }}>
            <label>Negative Prompt</label>
            <input
              className="input"
              type="text"
              value={params.negativePrompt}
              disabled={params.useWorkflowDefaults}
              onChange={(event) =>
                setParams((prev) => ({
                  ...prev,
                  negativePrompt: event.target.value,
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
            <label>CFG Scale</label>
            <input
              className="input"
              type="number"
              step={0.1}
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
              onChange={(event) =>
                setParams((prev) => ({
                  ...prev,
                  scheduler: event.target.value,
                }))
              }
            >
              <option value="normal">normal</option>
              <option value="karras">karras</option>
              <option value="simple">simple</option>
            </select>
          </div>
          <div>
            <label>Denoise</label>
            <input
              className="input"
              type="number"
              step={0.05}
              min={0}
              max={1}
              value={params.denoise}
              disabled={params.useWorkflowDefaults}
              onChange={(event) =>
                setParams((prev) => ({
                  ...prev,
                  denoise: Number(event.target.value),
                }))
              }
            />
          </div>
          <div>
            <label>Mask Strength</label>
            <input
              className="input"
              type="number"
              step={0.05}
              min={0}
              max={1}
              value={params.maskStrength}
              onChange={(event) =>
                setParams((prev) => ({
                  ...prev,
                  maskStrength: Number(event.target.value),
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
