export function getDisplayModelName(modelId: string): string {
  const name = modelId.split('/').at(-1) ?? modelId;
  return name.replace(/all-MiniLM-L\d+-v\d+/, 'MiniLM');
}
