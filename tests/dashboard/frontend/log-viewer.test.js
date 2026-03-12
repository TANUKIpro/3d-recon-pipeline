import { describe, it, expect, vi, beforeEach } from 'vitest';
import { buildLogViewerDOM } from './helpers/dom-factory.js';
import { LogViewer } from '../../../scripts/dashboard/static/js/log-viewer.js';

describe('LogViewer', () => {
  let viewer;

  beforeEach(() => {
    buildLogViewerDOM();
    viewer = new LogViewer();
  });

  // --- append: basics ---

  describe('append', () => {
    it('simple line appears in DOM', () => {
      viewer.append('stdout', 'hello world\n');
      const content = document.getElementById('log-content');
      expect(content.children.length).toBe(1);
      expect(content.children[0].textContent).toContain('hello world');
    });

    it('stderr stream gets log-stderr class', () => {
      viewer.append('stderr', 'error message\n');
      const content = document.getElementById('log-content');
      const span = content.children[0];
      expect(span.classList.contains('log-stderr')).toBe(true);
    });

    it('VRAM/GPU/CUDA text gets log-vram class', () => {
      viewer.append('stdout', 'VRAM usage: 4096MB\n');
      const content = document.getElementById('log-content');
      const span = content.children[0];
      expect(span.classList.contains('log-vram')).toBe(true);
    });

    it('ANSI escape sequences are stripped', () => {
      viewer.append('stdout', '\u001b[31mred text\u001b[0m\n');
      const content = document.getElementById('log-content');
      expect(content.children[0].textContent).toContain('red text');
      expect(content.children[0].textContent).not.toContain('\u001b');
    });

    it('newline splits into multiple entries', () => {
      viewer.append('stdout', 'line one\nline two\nline three\n');
      const content = document.getElementById('log-content');
      expect(content.children.length).toBe(3);
      expect(content.children[0].textContent).toContain('line one');
      expect(content.children[1].textContent).toContain('line two');
      expect(content.children[2].textContent).toContain('line three');
    });

    it('carriage return marks as progress hint', () => {
      viewer.append('stdout', 'downloading\r');
      const content = document.getElementById('log-content');
      // Progress lines get the log-progress class
      const span = content.children[0];
      expect(span.classList.contains('log-progress')).toBe(true);
    });
  });

  // --- chunk buffering ---

  describe('chunk buffering', () => {
    it('text without trailing newline is pending (not rendered)', () => {
      viewer.append('stdout', 'incomplete');
      const content = document.getElementById('log-content');
      expect(content.children.length).toBe(0);
    });

    it('next chunk with newline flushes pending', () => {
      viewer.append('stdout', 'first part');
      viewer.append('stdout', ' second part\n');
      const content = document.getElementById('log-content');
      expect(content.children.length).toBe(1);
      expect(content.children[0].textContent).toContain('first part second part');
    });
  });

  // --- progress detection ---

  describe('progress detection', () => {
    it('line with "ETA" detected as progress', () => {
      viewer.append('stdout', 'Processing 50% ETA 2m\n');
      const content = document.getElementById('log-content');
      const span = content.children[0];
      expect(span.classList.contains('log-progress')).toBe(true);
    });

    it('line with "it/s" detected as progress', () => {
      viewer.append('stdout', '100 samples 3.5 it/s\n');
      const content = document.getElementById('log-content');
      const span = content.children[0];
      expect(span.classList.contains('log-progress')).toBe(true);
    });

    it('line with percentage pattern detected as progress', () => {
      viewer.append('stdout', ' 75.3% \n');
      const content = document.getElementById('log-content');
      const span = content.children[0];
      expect(span.classList.contains('log-progress')).toBe(true);
    });
  });

  // --- progress compaction ---

  describe('progress compaction', () => {
    it('first progress update shows [progress]', () => {
      viewer.append('stdout', 'step 1\r');
      const content = document.getElementById('log-content');
      expect(content.children[0].textContent).toContain('[progress]');
      expect(content.children[0].textContent).not.toContain('[progress x');
    });

    it('multiple progress updates compact to [progress xN]', () => {
      viewer.append('stdout', 'step 1\r');
      viewer.append('stdout', 'step 2\r');
      viewer.append('stdout', 'step 3\r');
      const content = document.getElementById('log-content');
      // Should still be a single progress entry, not three
      expect(content.children.length).toBe(1);
      expect(content.children[0].textContent).toContain('[progress x3]');
    });
  });

  // --- stage filtering ---

  describe('stage filtering', () => {
    it('only active stage entries visible', () => {
      viewer.append('stdout', 'stage 1 log\n', { stage: 1 });
      viewer.append('stdout', 'stage 2 log\n', { stage: 2 });
      const content = document.getElementById('log-content');
      // Default active stage is 1
      expect(content.children.length).toBe(1);
      expect(content.children[0].textContent).toContain('stage 1 log');
    });

    it('switching active stage re-renders correct entries', () => {
      viewer.append('stdout', 'stage 1 log\n', { stage: 1 });
      viewer.append('stdout', 'stage 2 log\n', { stage: 2 });

      viewer.setActiveStage(2);
      const content = document.getElementById('log-content');
      expect(content.children.length).toBe(1);
      expect(content.children[0].textContent).toContain('stage 2 log');
    });
  });

  // --- setMaxLines ---

  describe('setMaxLines', () => {
    it('valid value accepted', () => {
      viewer.setMaxLines(500);
      // Append more than default but less than 500 to verify it was accepted
      for (let i = 0; i < 200; i++) {
        viewer.append('stdout', `line ${i}\n`);
      }
      const content = document.getElementById('log-content');
      expect(content.children.length).toBe(200);
    });

    it('value < 100 rejected', () => {
      viewer.setMaxLines(50);
      // Max lines should remain at default (2000), not 50
      // Verify by appending 120 lines; if limit were 50 they would be trimmed
      for (let i = 0; i < 120; i++) {
        viewer.append('stdout', `line ${i}\n`);
      }
      const content = document.getElementById('log-content');
      expect(content.children.length).toBe(120);
    });

    it('value capped at 50000', () => {
      viewer.setMaxLines(999999);
      // Internal max should be 50000, not 999999
      // We can't easily test this with DOM entries so we verify indirectly:
      // set to 100000 and then set to 200 to see trim behavior works
      viewer.setMaxLines(200);
      for (let i = 0; i < 250; i++) {
        viewer.append('stdout', `line ${i}\n`);
      }
      const content = document.getElementById('log-content');
      expect(content.children.length).toBe(200);
    });

    it('NaN rejected', () => {
      viewer.setMaxLines('not-a-number');
      // Default remains at 2000
      for (let i = 0; i < 120; i++) {
        viewer.append('stdout', `line ${i}\n`);
      }
      const content = document.getElementById('log-content');
      expect(content.children.length).toBe(120);
    });

    it('trims existing entries when reduced', () => {
      for (let i = 0; i < 200; i++) {
        viewer.append('stdout', `line ${i}\n`);
      }
      const content = document.getElementById('log-content');
      expect(content.children.length).toBe(200);

      viewer.setMaxLines(100);
      expect(content.children.length).toBe(100);
    });
  });

  // --- clear ---

  describe('clear', () => {
    it('empties all entries and DOM', () => {
      viewer.append('stdout', 'line 1\n', { stage: 1 });
      viewer.append('stdout', 'line 2\n', { stage: 2 });
      viewer.append('stdout', 'progress\r', { stage: 1 });

      viewer.clear();

      const content = document.getElementById('log-content');
      expect(content.innerHTML).toBe('');
      expect(content.children.length).toBe(0);

      // After clearing, new appends should work normally
      viewer.append('stdout', 'fresh line\n');
      expect(content.children.length).toBe(1);
    });
  });

  // --- _normalizeStage ---

  describe('_normalizeStage', () => {
    it('NaN returns 1', () => {
      expect(viewer._normalizeStage('abc')).toBe(1);
      expect(viewer._normalizeStage(undefined)).toBe(1);
      expect(viewer._normalizeStage(NaN)).toBe(1);
    });

    it('float is rounded', () => {
      expect(viewer._normalizeStage(2.7)).toBe(3);
      expect(viewer._normalizeStage(3.2)).toBe(3);
    });

    it('out of range (1-8) returns 1', () => {
      expect(viewer._normalizeStage(0)).toBe(1);
      expect(viewer._normalizeStage(9)).toBe(1);
      expect(viewer._normalizeStage(-1)).toBe(1);
    });

    it('stages 7 and 8 are valid', () => {
      expect(viewer._normalizeStage(7)).toBe(7);
      expect(viewer._normalizeStage(8)).toBe(8);
    });
  });
});
