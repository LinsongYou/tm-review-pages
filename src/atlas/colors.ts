const DEFAULT_FALLBACK_RGB = '127, 176, 105';

function parseHexRgb(hex: string): [number, number, number] | null {
  const value = hex.replace('#', '');
  const normalized =
    value.length === 3 ? value.split('').map((part) => `${part}${part}`).join('') : value;

  if (normalized.length !== 6) return null;

  const r = Number.parseInt(normalized.slice(0, 2), 16);
  const g = Number.parseInt(normalized.slice(2, 4), 16);
  const b = Number.parseInt(normalized.slice(4, 6), 16);

  if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return null;

  return [r, g, b];
}

export function hexToRgba(
  hex: string,
  alpha: number,
  fallbackRgb = DEFAULT_FALLBACK_RGB,
): string {
  const rgb = parseHexRgb(hex);
  if (!rgb) return `rgba(${fallbackRgb}, ${alpha})`;
  return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
}

export function isLightHex(hex: string): boolean {
  const rgb = parseHexRgb(hex);
  if (!rgb) return false;
  return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) > 128;
}

export function blendHexColors(a: string, b: string, fallbackRgb = DEFAULT_FALLBACK_RGB): string {
  const rgbA = parseHexRgb(a);
  const rgbB = parseHexRgb(b);
  if (!rgbA || !rgbB) return fallbackRgb;

  const r = Math.round((rgbA[0] + rgbB[0]) / 2);
  const g = Math.round((rgbA[1] + rgbB[1]) / 2);
  const bl = Math.round((rgbA[2] + rgbB[2]) / 2);
  return `${r}, ${g}, ${bl}`;
}
