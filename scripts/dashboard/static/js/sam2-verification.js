/**
 * SAM2 verification strip — shows 10 evenly-spaced composite frames
 * after mask propagation, with approve/redo buttons.
 * Supports dual mask display (object + ground).
 */

export class SAM2Verification {
  constructor() {
    this._container = document.getElementById('sam2-verification');
    this._strip = document.getElementById('sam2-verification-strip');
    this._approveBtn = document.getElementById('sam2-approve');
    this._redoBtn = document.getElementById('sam2-redo');
    this._visible = false;

    this._bindEvents();
  }

  /**
   * Show the verification strip with 10 evenly-spaced frames.
   * @param {number} totalFrames - Total number of frames in the video
   * @param {boolean} hasGround - Whether ground masks exist
   */
  show(totalFrames, hasGround = false) {
    const indices = this._pickIndices(totalFrames, 10);
    this._strip.innerHTML = '';

    for (const idx of indices) {
      const frame = document.createElement('div');
      frame.className = 'verification-frame';

      const img = document.createElement('img');
      img.src = `/api/verification/frame/${idx}?t=${Date.now()}`;
      img.alt = `Frame ${idx}`;
      img.loading = 'lazy';

      frame.appendChild(img);

      // If ground masks exist, show a second row with ground overlay
      if (hasGround) {
        const groundImg = document.createElement('img');
        groundImg.src = `/api/verification/ground-frame/${idx}?t=${Date.now()}`;
        groundImg.alt = `Ground ${idx}`;
        groundImg.loading = 'lazy';
        groundImg.className = 'verification-ground-img';
        frame.appendChild(groundImg);
      }

      const label = document.createElement('span');
      label.textContent = `#${idx}`;

      frame.appendChild(label);
      this._strip.appendChild(frame);
    }

    this._container.style.display = 'block';
    this._approveBtn.disabled = false;
    this._redoBtn.disabled = false;
    this._visible = true;
  }

  hide() {
    this._container.style.display = 'none';
    this._strip.innerHTML = '';
    this._visible = false;
  }

  get visible() {
    return this._visible;
  }

  /**
   * Pick `count` evenly-spaced indices from [0, total-1].
   */
  _pickIndices(total, count) {
    if (total <= 0) return [];
    if (total <= count) {
      return Array.from({ length: total }, (_, i) => i);
    }
    const step = (total - 1) / (count - 1);
    return Array.from({ length: count }, (_, i) => Math.round(i * step));
  }

  _bindEvents() {
    this._approveBtn.addEventListener('click', async () => {
      this._approveBtn.disabled = true;
      this._redoBtn.disabled = true;
      this._approveBtn.textContent = 'Proceeding...';
      try {
        await fetch('/api/sam2/approve', { method: 'POST' });
      } catch (e) {
        console.error('Approve error:', e);
      }
      this.hide();
      this._approveBtn.textContent = 'Approve & Continue';
    });

    this._redoBtn.addEventListener('click', async () => {
      this._approveBtn.disabled = true;
      this._redoBtn.disabled = true;
      this._redoBtn.textContent = 'Restarting...';
      try {
        await fetch('/api/sam2/redo', { method: 'POST' });
      } catch (e) {
        console.error('Redo error:', e);
      }
      this.hide();
      this._redoBtn.textContent = 'Redo';
    });
  }
}
