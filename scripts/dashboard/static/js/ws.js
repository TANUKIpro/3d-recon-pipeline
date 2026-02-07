/**
 * WebSocket manager with auto-reconnect and message dispatch.
 */

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 15000;

export class WsManager {
  constructor() {
    this._ws = null;
    this._listeners = {};  // type → [callback, ...]
    this._reconnectDelay = RECONNECT_BASE_MS;
    this._reconnectTimer = null;
    this._intentionalClose = false;
  }

  connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws`;
    this._intentionalClose = false;

    try {
      this._ws = new WebSocket(url);
    } catch (e) {
      this._scheduleReconnect();
      return;
    }

    this._ws.onopen = () => {
      this._reconnectDelay = RECONNECT_BASE_MS;
      this._dispatch('_open', null);
    };

    this._ws.onclose = () => {
      this._dispatch('_close', null);
      if (!this._intentionalClose) {
        this._scheduleReconnect();
      }
    };

    this._ws.onerror = () => {};

    this._ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type) {
          this._dispatch(msg.type, msg);
        }
        // Also fire wildcard
        this._dispatch('*', msg);
      } catch (e) {
        console.warn('WS parse error:', e);
      }
    };
  }

  on(type, callback) {
    if (!this._listeners[type]) this._listeners[type] = [];
    this._listeners[type].push(callback);
  }

  off(type, callback) {
    const arr = this._listeners[type];
    if (arr) {
      const idx = arr.indexOf(callback);
      if (idx >= 0) arr.splice(idx, 1);
    }
  }

  close() {
    this._intentionalClose = true;
    if (this._reconnectTimer) clearTimeout(this._reconnectTimer);
    if (this._ws) this._ws.close();
  }

  _dispatch(type, msg) {
    const arr = this._listeners[type];
    if (arr) {
      for (const cb of arr) {
        try { cb(msg); } catch (e) { console.error('WS handler error:', e); }
      }
    }
  }

  _scheduleReconnect() {
    if (this._reconnectTimer) return;
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this._reconnectDelay = Math.min(this._reconnectDelay * 1.5, RECONNECT_MAX_MS);
      this.connect();
    }, this._reconnectDelay);
  }
}
