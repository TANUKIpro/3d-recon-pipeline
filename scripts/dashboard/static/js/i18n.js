/**
 * Minimal i18n system with dictionary-based translations.
 * Reads/writes language preference to localStorage.
 */

const STORAGE_KEY = 'im2pc:lang';
const DEFAULT_LANG = 'en';

const TRANSLATIONS = {
  // Header
  'header.title': { en: 'im2pc-pipeline', ja: 'im2pc-pipeline' },
  'header.subtitle': { en: 'RGB Video → Textured 3D Mesh', ja: 'RGB動画 → テクスチャ付き3Dメッシュ' },

  // Stage labels
  'stage.extract_frames': { en: 'Extract Frames', ja: 'フレーム抽出' },
  'stage.pi3x': { en: 'Pi3X', ja: 'Pi3X' },
  'stage.sam2': { en: 'SAM2', ja: 'SAM2' },
  'stage.denoise': { en: 'Denoise', ja: 'ノイズ除去' },
  'stage.learning_mesh': { en: 'Learning Mesh', ja: 'Learning Mesh' },
  'stage.classical_mesh': { en: 'Classical Mesh', ja: 'Classical Mesh' },
  'stage.wrap': { en: 'Wrap', ja: 'ラップ' },
  'stage.repair': { en: 'Repair', ja: '修復' },
  'stage.texture': { en: 'Texture', ja: 'テクスチャ' },

  // Section titles
  'config.title': { en: 'Configuration', ja: '設定' },
  'log.title': { en: 'Log', ja: 'ログ' },
  'checkpoint.title': { en: 'Checkpoint', ja: 'チェックポイント' },

  // Settings panel
  'settings.title': { en: 'Settings', ja: '設定' },
  'settings.theme': { en: 'Theme', ja: 'テーマ' },
  'settings.theme.dark': { en: 'Dark', ja: 'ダーク' },
  'settings.theme.light': { en: 'Light', ja: 'ライト' },
  'settings.language': { en: 'Language', ja: '言語' },
  'settings.log': { en: 'Log Settings', ja: 'ログ設定' },
  'settings.log.autoscroll': { en: 'Auto-scroll', ja: '自動スクロール' },
  'settings.log.maxlines': { en: 'Max lines per stage', ja: 'ステージごとの最大行数' },
  'settings.close': { en: 'Close', ja: '閉じる' },
};

export class I18n {
  constructor() {
    this._lang = localStorage.getItem(STORAGE_KEY) || DEFAULT_LANG;
  }

  get lang() {
    return this._lang;
  }

  /**
   * Get translated string for key. Supports {0}, {1} placeholders.
   */
  t(key, ...params) {
    const entry = TRANSLATIONS[key];
    if (!entry) return key;
    let text = entry[this._lang] || entry[DEFAULT_LANG] || key;
    for (let i = 0; i < params.length; i++) {
      text = text.replace(`{${i}}`, params[i]);
    }
    return text;
  }

  /**
   * Apply translations to all elements with [data-i18n] attribute.
   */
  apply() {
    const els = document.querySelectorAll('[data-i18n]');
    for (const el of els) {
      const key = el.getAttribute('data-i18n');
      if (key) {
        el.textContent = this.t(key);
      }
    }
  }

  /**
   * Switch language and re-apply.
   */
  setLang(lang) {
    if (lang === this._lang) return;
    this._lang = lang;
    localStorage.setItem(STORAGE_KEY, lang);
    this.apply();
  }
}
