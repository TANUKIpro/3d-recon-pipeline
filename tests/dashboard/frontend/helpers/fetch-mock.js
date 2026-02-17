/**
 * Fetch mock utility using a Map of route → response.
 *
 * Usage:
 *   const { installFetch, addRoute, clearRoutes } = useFetchMock();
 *   installFetch();
 *   addRoute('/api/foo', { ok: true, json: async () => ({ result: 42 }) });
 */
import { vi } from 'vitest';

export function useFetchMock() {
  const routes = new Map();
  let originalFetch;

  function mockFetch(url, options = {}) {
    const urlStr = typeof url === 'string' ? url : url.toString();
    // Match against path part only (strip query string for matching)
    const path = urlStr.split('?')[0];

    for (const [pattern, handler] of routes) {
      if (path === pattern || urlStr === pattern || path.startsWith(pattern)) {
        if (typeof handler === 'function') {
          return Promise.resolve(handler(urlStr, options));
        }
        return Promise.resolve(handler);
      }
    }

    // Default 404
    return Promise.resolve({
      ok: false,
      status: 404,
      json: async () => ({ error: 'Not found' }),
      text: async () => 'Not found',
      blob: async () => new Blob(),
    });
  }

  return {
    installFetch() {
      originalFetch = globalThis.fetch;
      globalThis.fetch = vi.fn(mockFetch);
    },

    restoreFetch() {
      if (originalFetch) {
        globalThis.fetch = originalFetch;
      }
    },

    addRoute(pattern, response) {
      routes.set(pattern, response);
    },

    clearRoutes() {
      routes.clear();
    },

    /** Create a standard JSON response */
    jsonResponse(data, status = 200) {
      return {
        ok: status >= 200 && status < 300,
        status,
        json: async () => data,
        text: async () => JSON.stringify(data),
        blob: async () => new Blob([JSON.stringify(data)], { type: 'application/json' }),
      };
    },

    /** Create a blob response (for image results) */
    blobResponse(content = '', type = 'image/png') {
      const blob = new Blob([content], { type });
      return {
        ok: true,
        status: 200,
        blob: async () => blob,
        json: async () => ({}),
        text: async () => content,
      };
    },
  };
}
