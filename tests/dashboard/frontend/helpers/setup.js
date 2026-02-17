/**
 * Global test setup — runs before each test file.
 */
import { beforeEach, vi } from 'vitest';

beforeEach(() => {
  localStorage.clear();
  document.body.innerHTML = '';
  document.documentElement.removeAttribute('data-theme');
  vi.restoreAllMocks();
});
