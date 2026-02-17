import { describe, it, expect, vi, beforeEach } from 'vitest';
import { buildSettingsPanelDOM } from './helpers/dom-factory.js';
import { SettingsPanel } from '../../../scripts/dashboard/static/js/settings-panel.js';

function createMockI18n(lang = 'en') {
  return {
    lang,
    t: (key) => key,
    setLang: vi.fn(),
  };
}

describe('SettingsPanel', () => {
  let panel;
  let i18n;

  beforeEach(() => {
    buildSettingsPanelDOM();
    i18n = createMockI18n();
  });

  // --- defaults ---

  describe('defaults (no localStorage)', () => {
    beforeEach(() => {
      panel = new SettingsPanel(i18n);
    });

    it('theme is "light"', () => {
      expect(panel.theme).toBe('light');
    });

    it('lang is "en"', () => {
      expect(panel.lang).toBe('en');
    });

    it('autoScroll is true', () => {
      expect(panel.autoScroll).toBe(true);
    });

    it('maxLines is 2000', () => {
      expect(panel.maxLines).toBe(2000);
    });

    it('autoAccept is false', () => {
      expect(panel.autoAccept).toBe(false);
    });
  });

  // --- localStorage reads ---

  describe('localStorage reads', () => {
    it('reads stored theme', () => {
      localStorage.setItem('clip2mesh:theme', 'dark');
      panel = new SettingsPanel(i18n);
      expect(panel.theme).toBe('dark');
    });

    it('reads stored lang', () => {
      localStorage.setItem('clip2mesh:lang', 'ja');
      panel = new SettingsPanel(i18n);
      expect(panel.lang).toBe('ja');
    });

    it('reads stored autoScroll', () => {
      localStorage.setItem('clip2mesh:log.autoScroll', 'false');
      panel = new SettingsPanel(i18n);
      expect(panel.autoScroll).toBe(false);
    });

    it('reads stored maxLines', () => {
      localStorage.setItem('clip2mesh:log.maxLines', '5000');
      panel = new SettingsPanel(i18n);
      expect(panel.maxLines).toBe(5000);
    });
  });

  // --- _readBool ---

  describe('_readBool', () => {
    beforeEach(() => {
      panel = new SettingsPanel(i18n);
    });

    it('key not in storage returns fallback', () => {
      expect(panel._readBool('nonexistent-key', true)).toBe(true);
      expect(panel._readBool('nonexistent-key', false)).toBe(false);
    });

    it('"false" returns false', () => {
      localStorage.setItem('test-bool', 'false');
      expect(panel._readBool('test-bool', true)).toBe(false);
    });

    it('"true" returns true', () => {
      localStorage.setItem('test-bool', 'true');
      expect(panel._readBool('test-bool', false)).toBe(true);
    });
  });

  // --- _readInt ---

  describe('_readInt', () => {
    beforeEach(() => {
      panel = new SettingsPanel(i18n);
    });

    it('key not in storage returns fallback', () => {
      expect(panel._readInt('nonexistent-key', 42, 0, 100)).toBe(42);
    });

    it('value clamped to min/max', () => {
      localStorage.setItem('test-int', '999');
      expect(panel._readInt('test-int', 50, 0, 100)).toBe(100);

      localStorage.setItem('test-int', '-10');
      expect(panel._readInt('test-int', 50, 0, 100)).toBe(0);
    });

    it('non-numeric returns fallback', () => {
      localStorage.setItem('test-int', 'abc');
      expect(panel._readInt('test-int', 42, 0, 100)).toBe(42);
    });
  });

  // --- open / close / toggle ---

  describe('open / close / toggle', () => {
    beforeEach(() => {
      panel = new SettingsPanel(i18n);
    });

    it('open adds "open" class, close removes it, toggle switches', () => {
      const overlay = panel._overlay;

      expect(overlay.classList.contains('open')).toBe(false);

      panel.open();
      expect(overlay.classList.contains('open')).toBe(true);

      panel.close();
      expect(overlay.classList.contains('open')).toBe(false);

      panel.toggle();
      expect(overlay.classList.contains('open')).toBe(true);

      panel.toggle();
      expect(overlay.classList.contains('open')).toBe(false);
    });
  });

  // --- theme change ---

  describe('theme change', () => {
    it('saves to localStorage and calls onThemeChanged callback', () => {
      panel = new SettingsPanel(i18n);
      const callback = vi.fn();
      panel.onThemeChanged = callback;

      // Find the dark theme button in the overlay
      const darkBtn = panel._overlay.querySelector('[data-value="dark"]');
      darkBtn.click();

      expect(localStorage.getItem('clip2mesh:theme')).toBe('dark');
      expect(callback).toHaveBeenCalledWith('dark');
      expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    });
  });

  // --- lang change ---

  describe('lang change', () => {
    it('saves to localStorage, calls i18n.setLang, and calls onLangChanged callback', () => {
      panel = new SettingsPanel(i18n);
      const callback = vi.fn();
      panel.onLangChanged = callback;

      // Find the Japanese language button in the overlay
      const jaBtn = panel._overlay.querySelector('[data-value="ja"]');
      jaBtn.click();

      expect(localStorage.getItem('clip2mesh:lang')).toBe('ja');
      expect(i18n.setLang).toHaveBeenCalledWith('ja');
      expect(callback).toHaveBeenCalledWith('ja');
    });
  });
});
