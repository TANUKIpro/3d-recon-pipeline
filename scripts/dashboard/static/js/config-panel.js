/**
 * Configuration panel: video selection and parameter inputs.
 */

export class ConfigPanel {
  constructor() {
    this._videoSelect = document.getElementById('video-select');
    this._startBtn = document.getElementById('btn-start');
    this._cancelBtn = document.getElementById('btn-cancel');

    this._inputs = {
      frame_interval: document.getElementById('cfg-frame-interval'),
      max_frames: document.getElementById('cfg-max-frames'),
      pixel_limit: document.getElementById('cfg-pixel-limit'),
      confidence_threshold: document.getElementById('cfg-conf-threshold'),
      edge_rtol: document.getElementById('cfg-edge-rtol'),
      sam2_model: document.getElementById('cfg-sam2-model'),
      diffcd_batch_size: document.getElementById('cfg-diffcd-batch'),
      diffcd_n_batches: document.getElementById('cfg-diffcd-nbatches'),
      diffcd_resolution: document.getElementById('cfg-diffcd-res'),
      texture_size: document.getElementById('cfg-texture-size'),
    };

    this.onStart = null;  // callback(config)
    this.onCancel = null; // callback()

    this._bindEvents();
    this._loadVideos();
  }

  setRunning(running) {
    this._startBtn.disabled = running;
    this._cancelBtn.disabled = !running;
    // Disable all inputs while running
    this._videoSelect.disabled = running;
    for (const inp of Object.values(this._inputs)) {
      inp.disabled = running;
    }
  }

  getConfig() {
    return {
      video_path: this._videoSelect.value,
      frame_interval: parseInt(this._inputs.frame_interval.value) || 10,
      max_frames: parseInt(this._inputs.max_frames.value) || 50,
      pixel_limit: parseInt(this._inputs.pixel_limit.value) || 255000,
      confidence_threshold: parseFloat(this._inputs.confidence_threshold.value) || 0.1,
      edge_rtol: parseFloat(this._inputs.edge_rtol.value) || 0.03,
      sam2_model: this._inputs.sam2_model.value,
      diffcd_batch_size: parseInt(this._inputs.diffcd_batch_size.value) || 3000,
      diffcd_n_batches: parseInt(this._inputs.diffcd_n_batches.value) || 25000,
      diffcd_resolution: parseInt(this._inputs.diffcd_resolution.value) || 384,
      texture_size: parseInt(this._inputs.texture_size.value) || 2048,
    };
  }

  async _loadVideos() {
    try {
      const res = await fetch('/api/pipeline/videos');
      const data = await res.json();
      this._videoSelect.innerHTML = '';

      if (data.videos.length === 0) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'No videos found in /data/input/';
        this._videoSelect.appendChild(opt);
        return;
      }

      for (const v of data.videos) {
        const opt = document.createElement('option');
        opt.value = v.path;
        opt.textContent = `${v.name} (${v.size_mb} MB)`;
        this._videoSelect.appendChild(opt);
      }
    } catch (e) {
      this._videoSelect.innerHTML = '<option value="">Error loading videos</option>';
    }
  }

  _bindEvents() {
    this._startBtn.addEventListener('click', () => {
      if (this.onStart) {
        const cfg = this.getConfig();
        if (!cfg.video_path) {
          alert('Please select a video file.');
          return;
        }
        this.onStart(cfg);
      }
    });

    this._cancelBtn.addEventListener('click', () => {
      if (this.onCancel) this.onCancel();
    });
  }
}
