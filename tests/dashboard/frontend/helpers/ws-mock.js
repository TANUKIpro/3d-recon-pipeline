/**
 * MockWebSocket class for testing WsManager.
 *
 * Usage:
 *   installMockWebSocket();
 *   const ws = new WsManager();
 *   ws.connect();
 *   const mock = MockWebSocket.lastInstance;
 *   mock.simulateOpen();
 *   mock.simulateMessage({ type: 'log', text: 'hello' });
 *   mock.simulateClose();
 */
import { vi } from 'vitest';

export class MockWebSocket {
  static instances = [];
  static get lastInstance() {
    return MockWebSocket.instances[MockWebSocket.instances.length - 1] || null;
  }

  constructor(url) {
    this.url = url;
    this.readyState = 0; // CONNECTING
    this.onopen = null;
    this.onclose = null;
    this.onmessage = null;
    this.onerror = null;
    this.closeCalled = false;
    MockWebSocket.instances.push(this);
  }

  close() {
    this.closeCalled = true;
    this.readyState = 3; // CLOSED
    if (this.onclose) this.onclose();
  }

  send() {}

  simulateOpen() {
    this.readyState = 1; // OPEN
    if (this.onopen) this.onopen();
  }

  simulateMessage(data) {
    if (this.onmessage) {
      this.onmessage({ data: JSON.stringify(data) });
    }
  }

  simulateClose() {
    this.readyState = 3; // CLOSED
    if (this.onclose) this.onclose();
  }

  simulateError() {
    if (this.onerror) this.onerror(new Error('WebSocket error'));
  }
}

export function installMockWebSocket() {
  MockWebSocket.instances = [];
  globalThis.WebSocket = MockWebSocket;
}

export function cleanupMockWebSocket() {
  MockWebSocket.instances = [];
}
