/**
 * Real-time log viewer with stage filtering and progress compaction.
 */

const STAGE_MIN = 1;
const STAGE_MAX = 6;
const DEFAULT_STAGE = 1;
const DEFAULT_MAX_LINES = 2000;
const ANSI_ESCAPE_RE = /\u001b\[[0-9;?]*[ -/]*[@-~]/g;
const PROGRESS_HINT_RE = /(\bETA\b|\bit\/s\b|^\s*\d{1,3}(?:\.\d+)?%\s*$|\[[=>.\-#\s]+\])/i;
const FRACTION_PROGRESS_RE = /\b\d+\s*\/\s*\d+\b/;

export class LogViewer {
  constructor() {
    this._container = document.getElementById('log-container');
    this._content = document.getElementById('log-content');
    this._clearBtn = document.getElementById('log-clear');
    this._autoScroll = true;
    this._maxLines = DEFAULT_MAX_LINES;
    this._activeStage = DEFAULT_STAGE;

    this._entriesByStage = new Map();       // stage -> [{...}]
    this._pendingByChannel = new Map();     // `${stage}|${stream}` -> dangling text
    this._progressByChannel = new Map();    // `${stage}|${stream}` -> entry object

    this._bindEvents();
  }

  append(stream, text, opts = {}) {
    if (!text) return;
    const stage = this._normalizeStage(opts.stage);
    const forceProgress = opts.progress === true;
    this._consumeChunk(stream, String(text), stage, forceProgress);
  }

  clear() {
    this._entriesByStage.clear();
    this._pendingByChannel.clear();
    this._progressByChannel.clear();
    this._content.innerHTML = '';
  }

  setAutoScroll(enabled) {
    this._autoScroll = enabled !== false;
  }

  setMaxLines(count) {
    const val = Number(count);
    if (!Number.isFinite(val) || val < 100) return;
    this._maxLines = Math.min(50000, val);
    // Trim existing entries
    for (const stage of this._entriesByStage.keys()) {
      this._trimStageEntries(stage);
    }
  }

  setActiveStage(stage) {
    const normalized = this._normalizeStage(stage);
    if (normalized === this._activeStage) return;
    this._activeStage = normalized;
    this._renderActiveStage();
  }

  _bindEvents() {
    this._clearBtn.addEventListener('click', () => this.clear());

    // Detect manual scroll (disable auto-scroll if user scrolled up)
    this._container.addEventListener('scroll', () => {
      const { scrollTop, scrollHeight, clientHeight } = this._container;
      this._autoScroll = scrollHeight - scrollTop - clientHeight < 40;
    });

    document.addEventListener('stage-activated', (event) => {
      const stage = Number(event?.detail?.stage);
      if (Number.isFinite(stage)) {
        this.setActiveStage(stage);
      }
    });
  }

  _normalizeStage(stage) {
    const n = Number(stage);
    if (!Number.isFinite(n)) return DEFAULT_STAGE;
    const rounded = Math.round(n);
    if (rounded < STAGE_MIN || rounded > STAGE_MAX) return DEFAULT_STAGE;
    return rounded;
  }

  _stageEntries(stage) {
    if (!this._entriesByStage.has(stage)) {
      this._entriesByStage.set(stage, []);
    }
    return this._entriesByStage.get(stage);
  }

  _consumeChunk(stream, chunk, stage, forceProgress) {
    const channelKey = `${stage}|${stream}`;
    let buffer = `${this._pendingByChannel.get(channelKey) || ''}${chunk}`;
    let start = 0;
    let sawCarriage = false;

    for (let i = 0; i < buffer.length; i++) {
      const ch = buffer[i];
      if (ch !== '\n' && ch !== '\r') continue;

      const line = buffer.slice(start, i);
      const isCarriage = ch === '\r';
      this._commitLine(stage, stream, line, forceProgress || isCarriage);

      if (isCarriage && buffer[i + 1] === '\n') {
        i += 1;
      }
      start = i + 1;
      sawCarriage = sawCarriage || isCarriage;
    }

    let tail = buffer.slice(start);

    // Progress bars usually update with "\r...<no newline>".
    if (sawCarriage && tail) {
      this._commitLine(stage, stream, tail, true);
      tail = '';
    }

    this._pendingByChannel.set(channelKey, tail);
  }

  _commitLine(stage, stream, rawLine, isProgressHint) {
    const cleanLine = rawLine.replace(ANSI_ESCAPE_RE, '');
    const text = cleanLine.trimEnd();
    if (!text && !isProgressHint) return;

    if (isProgressHint || this._looksLikeProgress(text)) {
      this._upsertProgress(stage, stream, text);
      return;
    }
    this._appendEntry(stage, {
      kind: 'line',
      stream,
      text,
    });
  }

  _looksLikeProgress(line) {
    if (!line) return false;
    const trimmed = line.trim();
    if (!trimmed) return false;
    if (PROGRESS_HINT_RE.test(trimmed)) return true;
    if (FRACTION_PROGRESS_RE.test(trimmed) && /progress|frame|step/i.test(trimmed)) return true;
    return false;
  }

  _upsertProgress(stage, stream, text) {
    const normalized = text.trim();
    if (!normalized) return;
    const channelKey = `${stage}|${stream}`;
    let entry = this._progressByChannel.get(channelKey);
    if (!entry) {
      entry = {
        kind: 'progress',
        stream,
        text: '',
        updates: 0,
        channelKey,
      };
      this._progressByChannel.set(channelKey, entry);
      this._appendEntry(stage, entry);
    }

    entry.updates += 1;
    entry.text = entry.updates > 1
      ? `[progress x${entry.updates}] ${normalized}`
      : `[progress] ${normalized}`;

    if (stage === this._activeStage && entry.el) {
      entry.el.textContent = `${entry.text}\n`;
      this._maybeScrollToBottom();
    }
  }

  _appendEntry(stage, entry) {
    const entries = this._stageEntries(stage);
    entries.push(entry);

    if (stage === this._activeStage) {
      const span = this._renderEntry(entry);
      this._content.appendChild(span);
      entry.el = span;
    }

    this._trimStageEntries(stage);

    if (stage === this._activeStage) {
      this._maybeScrollToBottom();
    }
  }

  _trimStageEntries(stage) {
    const entries = this._stageEntries(stage);
    while (entries.length > this._maxLines) {
      const removed = entries.shift();
      if (removed?.channelKey && this._progressByChannel.get(removed.channelKey) === removed) {
        this._progressByChannel.delete(removed.channelKey);
      }
    }

    if (stage === this._activeStage && this._content.childNodes.length > entries.length) {
      while (this._content.childNodes.length > entries.length) {
        this._content.removeChild(this._content.firstChild);
      }
    }
  }

  _renderActiveStage() {
    this._content.innerHTML = '';
    const entries = this._stageEntries(this._activeStage);
    for (const entry of entries) {
      const span = this._renderEntry(entry);
      this._content.appendChild(span);
      entry.el = span;
    }
    this._maybeScrollToBottom(true);
  }

  _renderEntry(entry) {
    const span = document.createElement('span');
    span.textContent = `${entry.text}\n`;

    if (entry.stream === 'stderr') {
      span.classList.add('log-stderr');
    } else if (entry.stream === 'warn') {
      span.classList.add('log-warn');
    } else if (/vram|gpu|cuda/i.test(entry.text)) {
      span.classList.add('log-vram');
    }
    if (entry.kind === 'progress') {
      span.classList.add('log-progress');
    }
    return span;
  }

  _maybeScrollToBottom(force = false) {
    if (force || this._autoScroll) {
      this._container.scrollTop = this._container.scrollHeight;
    }
  }
}
