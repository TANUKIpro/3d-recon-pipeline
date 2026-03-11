/**
 * SAM2 interactive canvas for click-based segmentation.
 *
 * Left-click = positive point, right-click = negative point.
 * Each click sends a request and receives back a mask overlay PNG.
 * Supports object and ground plane segmentation modes.
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

    // Mode toggle buttons
    this._modeBtnObject = document.getElementById('sam2-mode-object');
    this._modeBtnGround = document.getElementById('sam2-mode-ground');
    this._skipGroundBtn = document.getElementById('sam2-skip-ground');

    this._active = false;
    this._imgWidth = 0;
    this._imgHeight = 0;
    this._positiveCount = 0;
    this._negativeCount = 0;
    this._groundPositiveCount = 0;
    this._groundNegativeCount = 0;
    this._loading = false;
    this._mode = 'object'; // 'object' or 'ground'
    this._groundPhase = false;

    this._bindEvents();
  }

  activate(frameCount, width, height) {
    this._active = true;
    this._imgWidth = width;
    this._imgHeight = height;
    this._positiveCount = 0;
    this._negativeCount = 0;
    this._groundPositiveCount = 0;
    this._groundNegativeCount = 0;
    this._mode = 'object';
    this._groundPhase = false;
    this.frameCount = frameCount;

    this._placeholder.style.display = 'none';
    this._canvas.style.display = 'block';
    this._canvas.width = width;
    this._canvas.height = height;
    this._loading = true;
    this._canvas.style.opacity = '0.7';
    this._undoBtn.disabled = true;
    this._clearBtn.disabled = true;
    this._confirmBtn.disabled = true;
    this._confirmBtn.textContent = 'Confirm & Propagate';

    // Hide mode toggle and skip button until ground phase
    this._setModeToggleVisible(false);
    this._setSkipGroundVisible(false);

    // Load first frame, then enable interactions.
    this._loadFrame(0).finally(() => {
      if (!this._active) return;
      this._loading = false;
      this._canvas.style.opacity = '1';
      this._undoBtn.disabled = false;
      this._clearBtn.disabled = false;
      this._confirmBtn.disabled = false;
    });
    this._updateInfo();
  }

  deactivate() {
    this._active = false;
    this._loading = false;
    this._groundPhase = false;
    this._canvas.style.opacity = '1';
    this._undoBtn.disabled = true;
    this._clearBtn.disabled = true;
    this._confirmBtn.disabled = true;
    this._setModeToggleVisible(false);
    this._setSkipGroundVisible(false);
  }

  /** Called when sam2_ground_phase WS message arrives */
  enterGroundPhase() {
    this._active = true;      // Ensure active (may have been cleared by reconnect)
    this._loading = false;    // Ensure not loading
    this._groundPhase = true;
    this._mode = 'ground';
    this._setModeToggleVisible(true);
    this._setSkipGroundVisible(true);
    this._updateModeButtons();
    this._updateInfo();
    this._confirmBtn.textContent = 'Confirm Ground & Propagate';
    this._confirmBtn.disabled = false;
    this._undoBtn.disabled = false;
    this._clearBtn.disabled = false;
    // Switch backend mode
    this._switchMode('ground');
  }

  /** Called when ground phase is skipped or done */
  exitGroundPhase() {
    this._groundPhase = false;
    this._setModeToggleVisible(false);
    this._setSkipGroundVisible(false);
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

    // Mode toggle buttons
    this._modeBtnObject?.addEventListener('click', () => {
      if (this._mode !== 'object') {
        this._mode = 'object';
        this._updateModeButtons();
        this._updateInfo();
        this._switchMode('object');
      }
    });
    this._modeBtnGround?.addEventListener('click', () => {
      if (this._mode !== 'ground') {
        this._mode = 'ground';
        this._updateModeButtons();
        this._updateInfo();
        this._switchMode('ground');
      }
    });

    // Skip ground button
    this._skipGroundBtn?.addEventListener('click', async () => {
      this._skipGroundBtn.disabled = true;
      try {
        await fetch('/api/sam2/skip-ground', { method: 'POST' });
      } catch (e) {
        console.error('Skip ground error:', e);
      }
    });
  }

  async _switchMode(mode) {
    try {
      await fetch('/api/sam2/mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      });
    } catch (e) {
      console.error('Mode switch error:', e);
    }
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

    if (this._mode === 'ground') {
      if (label === 1) this._groundPositiveCount++;
      else this._groundNegativeCount++;
    } else {
      if (label === 1) this._positiveCount++;
      else this._negativeCount++;
    }
    this._updateInfo();

    await this._postClickAndDraw('/api/sam2/click', {
      x: normX,
      y: normY,
      label: label,
    });
  }

  async _undo() {
    if (!this._active || this._loading) return;
    await this._postClickAndDraw('/api/sam2/undo');
    if (this._mode === 'ground') {
      this._groundPositiveCount = Math.max(0, this._groundPositiveCount - 1);
    } else {
      this._positiveCount = Math.max(0, this._positiveCount - 1);
    }
    this._updateInfo();
  }

  async _clear() {
    if (!this._active || this._loading) return;
    if (this._mode === 'ground') {
      this._groundPositiveCount = 0;
      this._groundNegativeCount = 0;
    } else {
      this._positiveCount = 0;
      this._negativeCount = 0;
    }
    this._updateInfo();
    await this._postClickAndDraw('/api/sam2/clear');
  }

  async _confirm() {
    if (!this._active) return;
    this._confirmBtn.disabled = true;
    const prevText = this._confirmBtn.textContent;
    this._confirmBtn.textContent = 'Propagating...';
    try {
      const res = await fetch('/api/sam2/confirm', { method: 'POST' });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        console.error('Confirm failed:', data.error || res.status);
        this._confirmBtn.disabled = false;
        this._confirmBtn.textContent = prevText;
      }
    } catch (e) {
      console.error('Confirm error:', e);
      this._confirmBtn.disabled = false;
      this._confirmBtn.textContent = prevText;
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
    const objInfo = `Object: ${this._positiveCount}+ ${this._negativeCount}-`;
    const groundInfo = `Ground: ${this._groundPositiveCount}+ ${this._groundNegativeCount}-`;
    const modeIndicator = this._groundPhase ? ` [${this._mode}]` : '';
    if (this._groundPhase || this._groundPositiveCount > 0 || this._groundNegativeCount > 0) {
      this._clickInfo.textContent = `${objInfo} | ${groundInfo}${modeIndicator}`;
    } else {
      this._clickInfo.textContent = `Positive: ${this._positiveCount}  Negative: ${this._negativeCount}`;
    }
  }

  _updateModeButtons() {
    if (this._modeBtnObject) {
      this._modeBtnObject.classList.toggle('active', this._mode === 'object');
    }
    if (this._modeBtnGround) {
      this._modeBtnGround.classList.toggle('active', this._mode === 'ground');
    }
  }

  _setModeToggleVisible(visible) {
    const container = document.getElementById('sam2-mode-toggle');
    if (container) {
      container.style.display = visible ? '' : 'none';
    }
  }

  _setSkipGroundVisible(visible) {
    if (this._skipGroundBtn) {
      this._skipGroundBtn.style.display = visible ? '' : 'none';
      this._skipGroundBtn.disabled = false;
    }
  }
}
