/**
 * SAM2 interactive canvas for click-based segmentation.
 *
 * Left-click = positive point, right-click = negative point.
 * Each click sends a request and receives back a mask overlay PNG.
 */

export class SAM2Canvas {
  constructor() {
    this._canvas = document.getElementById('sam2-canvas');
    this._ctx = this._canvas.getContext('2d');
    this._placeholder = document.getElementById('sam2-placeholder');
    this._undoBtn = document.getElementById('sam2-undo');
    this._clearBtn = document.getElementById('sam2-clear');
    this._confirmBtn = document.getElementById('sam2-confirm');
    this._clickInfo = document.getElementById('sam2-click-info');

    this._active = false;
    this._imgWidth = 0;
    this._imgHeight = 0;
    this._positiveCount = 0;
    this._negativeCount = 0;
    this._loading = false;

    this._bindEvents();
  }

  activate(frameCount, width, height) {
    this._active = true;
    this._imgWidth = width;
    this._imgHeight = height;
    this._positiveCount = 0;
    this._negativeCount = 0;
    this.frameCount = frameCount;

    this._placeholder.style.display = 'none';
    this._canvas.style.display = 'block';
    this._undoBtn.disabled = false;
    this._clearBtn.disabled = false;
    this._confirmBtn.disabled = false;
    this._confirmBtn.textContent = 'Confirm & Propagate';

    // Load first frame
    this._loadFrame(0);
    this._updateInfo();
  }

  deactivate() {
    this._active = false;
    this._undoBtn.disabled = true;
    this._clearBtn.disabled = true;
    this._confirmBtn.disabled = true;
  }

  async _loadFrame(idx) {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    return new Promise((resolve) => {
      img.onload = () => {
        this._canvas.width = img.naturalWidth;
        this._canvas.height = img.naturalHeight;
        this._ctx.drawImage(img, 0, 0);
        resolve();
      };
      img.onerror = resolve;
      img.src = `/api/sam2/frame/${idx}?t=${Date.now()}`;
    });
  }

  _bindEvents() {
    // Left click = positive, right click = negative
    this._canvas.addEventListener('click', (e) => this._handleClick(e, 1));
    this._canvas.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      this._handleClick(e, 0);
    });

    this._undoBtn.addEventListener('click', () => this._undo());
    this._clearBtn.addEventListener('click', () => this._clear());
    this._confirmBtn.addEventListener('click', () => this._confirm());
  }

  async _handleClick(e, label) {
    if (!this._active || this._loading) return;

    const rect = this._canvas.getBoundingClientRect();
    const scaleX = this._canvas.width / rect.width;
    const scaleY = this._canvas.height / rect.height;
    const canvasX = (e.clientX - rect.left) * scaleX;
    const canvasY = (e.clientY - rect.top) * scaleY;
    const normX = canvasX / this._canvas.width;
    const normY = canvasY / this._canvas.height;

    if (label === 1) this._positiveCount++;
    else this._negativeCount++;
    this._updateInfo();

    await this._postClickAndDraw('/api/sam2/click', {
      x: normX,
      y: normY,
      label: label,
    });
  }

  async _undo() {
    if (!this._active || this._loading) return;
    if (this._positiveCount + this._negativeCount > 0) {
      // We don't track which was last, just decrement total
      // The backend handles the actual undo
    }
    await this._postClickAndDraw('/api/sam2/undo');
    // Refresh counts from a rough estimate
    this._positiveCount = Math.max(0, this._positiveCount - 1);
    this._updateInfo();
  }

  async _clear() {
    if (!this._active || this._loading) return;
    this._positiveCount = 0;
    this._negativeCount = 0;
    this._updateInfo();
    await this._postClickAndDraw('/api/sam2/clear');
  }

  async _confirm() {
    if (!this._active) return;
    this._confirmBtn.disabled = true;
    this._confirmBtn.textContent = 'Propagating...';
    try {
      const res = await fetch('/api/sam2/confirm', { method: 'POST' });
      if (!res.ok) {
        const data = await res.json();
        console.error('Confirm failed:', data.error);
      }
    } catch (e) {
      console.error('Confirm error:', e);
    }
  }

  async _postClickAndDraw(url, body) {
    this._loading = true;
    this._canvas.style.opacity = '0.7';
    try {
      const options = { method: 'POST' };
      if (body) {
        options.headers = { 'Content-Type': 'application/json' };
        options.body = JSON.stringify(body);
      }
      const res = await fetch(url, options);
      if (res.ok) {
        const blob = await res.blob();
        const img = new Image();
        await new Promise((resolve, reject) => {
          img.onload = resolve;
          img.onerror = reject;
          img.src = URL.createObjectURL(blob);
        });
        this._canvas.width = img.naturalWidth;
        this._canvas.height = img.naturalHeight;
        this._ctx.drawImage(img, 0, 0);
        URL.revokeObjectURL(img.src);
      }
    } catch (e) {
      console.error('SAM2 request error:', e);
    } finally {
      this._loading = false;
      this._canvas.style.opacity = '1';
    }
  }

  _updateInfo() {
    this._clickInfo.textContent =
      `Positive: ${this._positiveCount}  Negative: ${this._negativeCount}`;
  }
}
