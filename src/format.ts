export function getDisplayModelName(modelId: string): string {
  return modelId.split('/').at(-1) ?? modelId;
}

export function formatPercent(value: number): string {
  if (!Number.isFinite(value)) {
    return '--';
  }

  const percent = value * 100;
  if (percent >= 10) {
    return `${Math.round(percent)}%`;
  }

  return `${percent.toFixed(1)}%`;
}
