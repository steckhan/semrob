export type OddFactor = { id: string; label: string };
export type OddDimension = { key: string; label: string; factors: OddFactor[] };
export type OddCatalog = {
  domain: string;
  dimensions: OddDimension[];
};

/**
 * Builds a comma-separated prompt string from selected factor IDs.
 * Factors are ordered by dimension (Actors → Activities → Environment → Sensors).
 */
export function buildPromptFromSelections(
  catalog: OddCatalog,
  selectedIds: Set<string>,
): string {
  const labels: string[] = [];
  for (const dimension of catalog.dimensions) {
    for (const factor of dimension.factors) {
      if (selectedIds.has(factor.id)) {
        labels.push(factor.label);
      }
    }
  }
  return labels.join(", ");
}
