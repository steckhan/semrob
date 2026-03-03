"use client";

type Props = {
  domain: string;
  onDomainChange: (domain: string) => void;
  onGenerate: () => void;
  isGenerating: boolean;
};

export default function OddDomainCard({
  domain,
  onDomainChange,
  onGenerate,
  isGenerating,
}: Props) {
  return (
    <div className="card">
      <div className="card-header">ODD Domain</div>
      <div className="card-body">
        <div>
          <label>Domain Description</label>
          <input
            className="input"
            type="text"
            value={domain}
            onChange={(e) => onDomainChange(e.target.value)}
            placeholder="e.g. hand detection for dimension saws"
            disabled={isGenerating}
          />
        </div>
        <button
          className="btn btn-outline btn-full btn-sm"
          onClick={onGenerate}
          disabled={!domain.trim() || isGenerating}
        >
          {isGenerating ? "Generating…" : "Generate Factors"}
        </button>
        <p className="hint">
          Uses an LLM to derive ODD factors for Actors, Activities, Environment, and Sensors.
        </p>
      </div>
    </div>
  );
}
