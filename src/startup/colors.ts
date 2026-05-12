const DEFAULT_FALLBACK_RGB = '127, 176, 105';

export function hexToRgba(
  hex: string,
  alpha: number,
  fallbackRgb = DEFAULT_FALLBACK_RGB,
): string {
  const value = hex.replace('#', '');
  const normalized =
    value.length === 3 ? value.split('').map((part) => `${part}${part}`).join('') : value;

  if (normalized.length !== 6) {
    return `rgba(${fallbackRgb}, ${alpha})`;
  }

  const red = Number.parseInt(normalized.slice(0, 2), 16);
  const green = Number.parseInt(normalized.slice(2, 4), 16);
  const blue = Number.parseInt(normalized.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}
