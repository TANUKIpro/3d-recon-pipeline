import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { buildI18nDOM } from './helpers/dom-factory.js';

/**
 * The TRANSLATIONS dict inside i18n.js is not exported and no current entries
 * contain {0}/{1} placeholders. To test the replacement logic we mock the
 * module and inject a test entry ('test.tpl') alongside the real translations
 * (re-exported via vi.importActual).
 */
vi.mock('../../../scripts/dashboard/static/js/i18n.js', async (importOriginal) => {
  const original = await importOriginal();

  // Create a subclass that injects a test translation with placeholders
  class TestableI18n extends original.I18n {
    t(key, ...params) {
      // Inject a fake translation entry for placeholder testing
      if (key === 'test.tpl') {
        let text = this._lang === 'ja'
          ? '{0}が{1}個処理しました'
          : '{0} processed {1} items';
        for (let i = 0; i < params.length; i++) {
          text = text.replace(`{${i}}`, params[i]);
        }
        return text;
      }
      if (key === 'test.tpl3') {
        let text = '{0} has {1} and {2}';
        for (let i = 0; i < params.length; i++) {
          text = text.replace(`{${i}}`, params[i]);
        }
        return text;
      }
      return super.t(key, ...params);
    }
  }

  return { ...original, I18n: TestableI18n };
});

const { I18n } = await import('../../../scripts/dashboard/static/js/i18n.js');

describe('I18n', () => {
  describe('constructor', () => {
    it('defaults to "en" when localStorage has no lang', () => {
      const i18n = new I18n();
      expect(i18n.lang).toBe('en');
    });

    it('reads language from localStorage', () => {
      localStorage.setItem('clip2mesh:lang', 'ja');
      const i18n = new I18n();
      expect(i18n.lang).toBe('ja');
    });
  });

  describe('t(key)', () => {
    it('returns English translation for known key', () => {
      const i18n = new I18n();
      expect(i18n.t('header.subtitle')).toBe('RGB Video \u2192 Textured 3D Mesh');
    });

    it('returns Japanese translation when lang is "ja"', () => {
      localStorage.setItem('clip2mesh:lang', 'ja');
      const i18n = new I18n();
      expect(i18n.t('header.subtitle')).toBe('RGB\u52d5\u753b \u2192 \u30c6\u30af\u30b9\u30c1\u30e3\u4ed8\u304d3D\u30e1\u30c3\u30b7\u30e5');
    });

    it('returns the key itself for unknown keys', () => {
      const i18n = new I18n();
      expect(i18n.t('nonexistent.key')).toBe('nonexistent.key');
    });
  });

  describe('t(key, ...params)', () => {
    it('replaces {0} and {1} placeholders with provided params', () => {
      const i18n = new I18n();
      const result = i18n.t('test.tpl', 'Stage1', 42);
      expect(result).toBe('Stage1 processed 42 items');
    });

    it('leaves unreplaced placeholders when fewer params than placeholders', () => {
      const i18n = new I18n();
      const result = i18n.t('test.tpl3', 'only-first');
      expect(result).toBe('only-first has {1} and {2}');
    });
  });

  describe('apply()', () => {
    it('updates textContent of all [data-i18n] elements', () => {
      buildI18nDOM();
      localStorage.setItem('clip2mesh:lang', 'ja');
      const i18n = new I18n();
      i18n.apply();

      const h1 = document.querySelector('[data-i18n="header.title"]');
      const p = document.querySelector('[data-i18n="header.subtitle"]');
      const span = document.querySelector('[data-i18n="stage.extract_frames"]');

      expect(h1.textContent).toBe('clip2mesh');
      expect(p.textContent).toBe('RGB\u52d5\u753b \u2192 \u30c6\u30af\u30b9\u30c1\u30e3\u4ed8\u304d3D\u30e1\u30c3\u30b7\u30e5');
      expect(span.textContent).toBe('\u30d5\u30ec\u30fc\u30e0\u62bd\u51fa');
    });

    it('skips elements with empty data-i18n attribute', () => {
      document.body.innerHTML = `
        <span data-i18n="">original text</span>
        <span data-i18n="header.title">clip2mesh</span>
      `;
      const i18n = new I18n();
      i18n.apply();

      const empty = document.querySelector('[data-i18n=""]');
      expect(empty.textContent).toBe('original text');
    });
  });

  describe('setLang()', () => {
    it('updates localStorage and re-translates DOM', () => {
      buildI18nDOM();
      const i18n = new I18n();

      i18n.setLang('ja');

      expect(i18n.lang).toBe('ja');
      expect(localStorage.getItem('clip2mesh:lang')).toBe('ja');

      const p = document.querySelector('[data-i18n="header.subtitle"]');
      expect(p.textContent).toBe('RGB\u52d5\u753b \u2192 \u30c6\u30af\u30b9\u30c1\u30e3\u4ed8\u304d3D\u30e1\u30c3\u30b7\u30e5');
    });

    it('is a no-op when setting the same language', () => {
      buildI18nDOM();
      const i18n = new I18n();
      const applySpy = vi.spyOn(i18n, 'apply');

      i18n.setLang('en'); // same as default

      expect(applySpy).not.toHaveBeenCalled();
    });
  });

  describe('lang getter', () => {
    it('returns the current language', () => {
      const i18n = new I18n();
      expect(i18n.lang).toBe('en');

      i18n.setLang('ja');
      expect(i18n.lang).toBe('ja');
    });
  });
});
