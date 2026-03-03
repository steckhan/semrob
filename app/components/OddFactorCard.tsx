"use client";

import { useState } from "react";

import type { OddCatalog } from "@/lib/oddCatalog";

type Props = {
  catalog: OddCatalog | null;
  selectedFactorIds: Set<string>;
  onSelectionChange: (ids: Set<string>) => void;
  disabled?: boolean;
};

export default function OddFactorCard({
  catalog,
  selectedFactorIds,
  onSelectionChange,
  disabled,
}: Props) {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  if (!catalog) {
    return (
      <div className="card">
        <div className="card-header">ODD Factors</div>
        <div className="card-body">
          <p className="hint" style={{ textAlign: "center", padding: "8px 0" }}>
            Enter a domain above and click Generate to get started.
          </p>
        </div>
      </div>
    );
  }

  const totalSelected = selectedFactorIds.size;

  const toggleFactor = (id: string) => {
    if (disabled) return;
    const next = new Set(selectedFactorIds);
    if (next.has(id)) {
      next.delete(id);
    } else {
      next.add(id);
    }
    onSelectionChange(next);
  };

  const clearAll = () => {
    onSelectionChange(new Set());
  };

  const toggleCollapse = (key: string) => {
    setCollapsed((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div className="card">
      <div className="card-header">
        <span>ODD Factors</span>
        {totalSelected > 0 && (
          <button
            className="odd-clear-btn"
            onClick={clearAll}
            disabled={disabled}
          >
            Clear ({totalSelected})
          </button>
        )}
      </div>
      <div className="card-body" style={{ gap: 0 }}>
        {catalog.dimensions.map((dim) => {
          const isCollapsed = collapsed[dim.key] ?? false;
          const selectedCount = dim.factors.filter((f) =>
            selectedFactorIds.has(f.id),
          ).length;

          return (
            <div key={dim.key} className="odd-dimension">
              <button
                className="odd-dimension-header"
                onClick={() => toggleCollapse(dim.key)}
              >
                <span className="odd-dimension-label">{dim.label}</span>
                {selectedCount > 0 && (
                  <span className="odd-dimension-count">{selectedCount}</span>
                )}
                <span className={`odd-chevron${isCollapsed ? " collapsed" : ""}`}>
                  ▾
                </span>
              </button>
              {!isCollapsed && (
                <div className="odd-chips">
                  {dim.factors.map((factor) => (
                    <button
                      key={factor.id}
                      className={`odd-chip${selectedFactorIds.has(factor.id) ? " selected" : ""}`}
                      onClick={() => toggleFactor(factor.id)}
                      disabled={disabled}
                    >
                      {factor.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
