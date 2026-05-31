export function getDisplayModelName(modelId: string): string {
  return modelId.split('/').at(-1) ?? modelId;
}
