import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { MockWebSocket, installMockWebSocket } from './helpers/ws-mock.js';
import { WsManager } from '../../../scripts/dashboard/static/js/ws.js';

beforeEach(() => {
  installMockWebSocket();
});

describe('WsManager', () => {
  describe('connect()', () => {
    it('constructs ws:// URL when protocol is http:', () => {
      Object.defineProperty(window, 'location', {
        value: { protocol: 'http:', host: 'localhost:7860' },
        writable: true,
        configurable: true,
      });

      const ws = new WsManager();
      ws.connect();

      const mock = MockWebSocket.lastInstance;
      expect(mock.url).toBe('ws://localhost:7860/ws');
    });

    it('constructs wss:// URL when protocol is https:', () => {
      Object.defineProperty(window, 'location', {
        value: { protocol: 'https:', host: 'example.com' },
        writable: true,
        configurable: true,
      });

      const ws = new WsManager();
      ws.connect();

      const mock = MockWebSocket.lastInstance;
      expect(mock.url).toBe('wss://example.com/ws');
    });
  });

  describe('on() / off()', () => {
    it('registers listener and receives messages of matching type', () => {
      const ws = new WsManager();
      const handler = vi.fn();
      ws.on('log', handler);
      ws.connect();

      const mock = MockWebSocket.lastInstance;
      mock.simulateOpen();
      mock.simulateMessage({ type: 'log', text: 'hello' });

      expect(handler).toHaveBeenCalledWith({ type: 'log', text: 'hello' });
    });

    it('fires multiple listeners registered for the same type', () => {
      const ws = new WsManager();
      const handler1 = vi.fn();
      const handler2 = vi.fn();
      ws.on('log', handler1);
      ws.on('log', handler2);
      ws.connect();

      const mock = MockWebSocket.lastInstance;
      mock.simulateOpen();
      mock.simulateMessage({ type: 'log', text: 'data' });

      expect(handler1).toHaveBeenCalledTimes(1);
      expect(handler2).toHaveBeenCalledTimes(1);
    });

    it('removes a specific listener with off()', () => {
      const ws = new WsManager();
      const handler = vi.fn();
      ws.on('log', handler);
      ws.off('log', handler);
      ws.connect();

      const mock = MockWebSocket.lastInstance;
      mock.simulateOpen();
      mock.simulateMessage({ type: 'log', text: 'hello' });

      expect(handler).not.toHaveBeenCalled();
    });

    it('off() with non-existent listener does not throw', () => {
      const ws = new WsManager();
      const handler = vi.fn();

      expect(() => ws.off('log', handler)).not.toThrow();
      expect(() => ws.off('nonexistent', handler)).not.toThrow();
    });
  });

  describe('message dispatch', () => {
    it('dispatches JSON message to correct type handler', () => {
      const ws = new WsManager();
      const logHandler = vi.fn();
      const stageHandler = vi.fn();
      ws.on('log', logHandler);
      ws.on('stage', stageHandler);
      ws.connect();

      const mock = MockWebSocket.lastInstance;
      mock.simulateOpen();
      mock.simulateMessage({ type: 'log', text: 'a log line' });

      expect(logHandler).toHaveBeenCalledTimes(1);
      expect(stageHandler).not.toHaveBeenCalled();
    });

    it('wildcard "*" listener receives all messages', () => {
      const ws = new WsManager();
      const wildcard = vi.fn();
      ws.on('*', wildcard);
      ws.connect();

      const mock = MockWebSocket.lastInstance;
      mock.simulateOpen();
      mock.simulateMessage({ type: 'log', text: 'msg1' });
      mock.simulateMessage({ type: 'stage', name: 'pi3x' });

      expect(wildcard).toHaveBeenCalledTimes(2);
      expect(wildcard).toHaveBeenCalledWith({ type: 'log', text: 'msg1' });
      expect(wildcard).toHaveBeenCalledWith({ type: 'stage', name: 'pi3x' });
    });

    it('fires "_open" event on WebSocket open', () => {
      const ws = new WsManager();
      const openHandler = vi.fn();
      ws.on('_open', openHandler);
      ws.connect();

      const mock = MockWebSocket.lastInstance;
      mock.simulateOpen();

      expect(openHandler).toHaveBeenCalledWith(null);
    });

    it('fires "_close" event on WebSocket close', () => {
      const ws = new WsManager();
      const closeHandler = vi.fn();
      ws.on('_close', closeHandler);
      ws.connect();

      const mock = MockWebSocket.lastInstance;
      mock.simulateOpen();
      mock.simulateClose();

      expect(closeHandler).toHaveBeenCalledWith(null);
    });
  });

  describe('error isolation', () => {
    it('handler throwing does not crash dispatch of other handlers', () => {
      const ws = new WsManager();
      const badHandler = vi.fn(() => { throw new Error('boom'); });
      const goodHandler = vi.fn();
      ws.on('log', badHandler);
      ws.on('log', goodHandler);
      ws.connect();

      const mock = MockWebSocket.lastInstance;
      mock.simulateOpen();

      expect(() => {
        mock.simulateMessage({ type: 'log', text: 'test' });
      }).not.toThrow();

      expect(badHandler).toHaveBeenCalledTimes(1);
      expect(goodHandler).toHaveBeenCalledTimes(1);
    });
  });

  describe('invalid JSON', () => {
    it('logs console.warn and does not crash', () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const ws = new WsManager();
      ws.connect();

      const mock = MockWebSocket.lastInstance;
      mock.simulateOpen();

      // Manually trigger onmessage with invalid JSON
      expect(() => {
        mock.onmessage({ data: 'not-valid-json{{{' });
      }).not.toThrow();

      expect(warnSpy).toHaveBeenCalledTimes(1);
      expect(warnSpy.mock.calls[0][0]).toBe('WS parse error:');
    });
  });

  describe('close()', () => {
    it('sets intentionalClose and calls ws.close()', () => {
      const ws = new WsManager();
      ws.connect();

      const mock = MockWebSocket.lastInstance;
      mock.simulateOpen();
      ws.close();

      expect(mock.closeCalled).toBe(true);
    });
  });

  describe('reconnect', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('schedules reconnect on unexpected close', () => {
      const ws = new WsManager();
      ws.connect();

      const mock = MockWebSocket.lastInstance;
      mock.simulateOpen();
      mock.simulateClose(); // unexpected close (intentionalClose is false)

      expect(MockWebSocket.instances).toHaveLength(1);

      vi.advanceTimersByTime(1000);

      // A new WebSocket instance should have been created
      expect(MockWebSocket.instances).toHaveLength(2);
    });

    it('backoff increases by 1.5x on each reconnect', () => {
      const ws = new WsManager();
      ws.connect();

      // First close: delay is 1000ms, after reconnect delay becomes 1500ms
      MockWebSocket.lastInstance.simulateOpen();
      MockWebSocket.lastInstance.simulateClose();
      expect(MockWebSocket.instances).toHaveLength(1);

      vi.advanceTimersByTime(1000); // triggers reconnect, delay updated to 1500
      expect(MockWebSocket.instances).toHaveLength(2);

      // Second close: delay is now 1500ms
      MockWebSocket.lastInstance.simulateClose();

      vi.advanceTimersByTime(1499);
      expect(MockWebSocket.instances).toHaveLength(2); // not yet

      vi.advanceTimersByTime(1);
      expect(MockWebSocket.instances).toHaveLength(3); // now reconnected

      // Third close: delay is now 2250ms
      MockWebSocket.lastInstance.simulateClose();

      vi.advanceTimersByTime(2249);
      expect(MockWebSocket.instances).toHaveLength(3);

      vi.advanceTimersByTime(1);
      expect(MockWebSocket.instances).toHaveLength(4);
    });

    it('close() prevents reconnection', () => {
      const ws = new WsManager();
      ws.connect();

      const mock = MockWebSocket.lastInstance;
      mock.simulateOpen();

      // Intentional close
      ws.close();

      expect(MockWebSocket.instances).toHaveLength(1);

      vi.advanceTimersByTime(20000); // well beyond any backoff

      // No new WebSocket instances should have been created
      expect(MockWebSocket.instances).toHaveLength(1);
    });
  });
});
