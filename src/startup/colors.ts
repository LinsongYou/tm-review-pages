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

export function isLightHex(hex: string): boolean {
  const value = hex.replace('#', '');
  const normalized = value.length === 3 ? value.split('').map((part) => `${part}${part}`).join('') : value;
  if (normalized.length !== 6) return false;
  const r = Number.parseInt(normalized.slice(0, 2), 16);
  const g = Number.parseInt(normalized.slice(2, 4), 16);
  const b = Number.parseInt(normalized.slice(4, 6), 16);
  return (0.299 * r + 0.587 * g + 0.114 * b) > 128;
}

export function blendHexColors(a: string, b: string, fallbackRgb = DEFAULT_FALLBACK_RGB): string {
  const parse = (hex: string) => {
    const value = hex.replace('#', '');
    const normalized =
      value.length === 3 ? value.split('').map((part) => `${part}${part}`).join('') : value;
    if (normalized.length !== 6) return null;
    return [
      Number.parseInt(normalized.slice(0, 2), 16),
      Number.parseInt(normalized.slice(2, 4), 16),
      Number.parseInt(normalized.slice(4, 6), 16),
    ] as const;
  };

  const rgbA = parse(a);
  const rgbB = parse(b);
  if (!rgbA || !rgbB) return fallbackRgb;

  const r = Math.round((rgbA[0] + rgbB[0]) / 2);
  const g = Math.round((rgbA[1] + rgbB[1]) / 2);
  const bl = Math.round((rgbA[2] + rgbB[2]) / 2);
  return `${r}, ${g}, ${bl}`;
}
