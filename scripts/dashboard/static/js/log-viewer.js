/**
 * Real-time log viewer with auto-scroll and VRAM highlighting.
 */

const MAX_LINES = 2000;

export class LogViewer {
  constructor() {
    this._container = document.getElementById('log-container');
    this._content = document.getElementById('log-content');
    this._clearBtn = document.getElementById('log-clear');
    this._lineCount = 0;
    this._autoScroll = true;

    this._bindEvents();
  }

  append(stream, text) {
    // Split into lines for per-line styling
    const lines = text.split('\n');
    for (const line of lines) {
      if (!line && lines.length > 1) continue; // skip empty splits
      const span = document.createElement('span');
      span.textContent = line + '\n';

      // Style based on content
      if (stream === 'stderr') {
        span.className = 'log-stderr';
      } else if (/vram|VRAM|GPU|cuda/i.test(line)) {
        span.className = 'log-vram';
      }

      this._content.appendChild(span);
      this._lineCount++;
    }

    // Trim old lines
    while (this._lineCount > MAX_LINES && this._content.firstChild) {
      this._content.removeChild(this._content.firstChild);
      this._lineCount--;
    }

    // Auto-scroll
    if (this._autoScroll) {
      this._container.scrollTop = this._container.scrollHeight;
    }
  }

  clear() {
    this._content.innerHTML = '';
    this._lineCount = 0;
  }

  _bindEvents() {
    this._clearBtn.addEventListener('click', () => this.clear());

    // Detect manual scroll (disable auto-scroll if user scrolled up)
    this._container.addEventListener('scroll', () => {
      const { scrollTop, scrollHeight, clientHeight } = this._container;
      this._autoScroll = scrollHeight - scrollTop - clientHeight < 40;
    });
  }
}
