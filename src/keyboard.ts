import type { KeyboardEvent as ReactKeyboardEvent } from 'react';

export function handleSelectKey(
  event: ReactKeyboardEvent<HTMLElement>,
  select: () => void,
): void {
  if (event.target !== event.currentTarget || (event.key !== 'Enter' && event.key !== ' ')) {
    return;
  }

  event.preventDefault();
  select();
}
