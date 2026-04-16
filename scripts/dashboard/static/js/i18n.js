/**
 * Minimal i18n system with dictionary-based translations.
 * Reads/writes language preference to localStorage.
 */

import { STORAGE_KEYS } from './constants.js';

const STORAGE_KEY = STORAGE_KEYS.lang;
const DEFAULT_LANG = 'en';

const TRANSLATIONS = {
  // Header
  'header.title': { en: 'clip2mesh', ja: 'clip2mesh' },
  'header.subtitle': { en: 'RGB Video → Textured 3D Mesh', ja: 'RGB動画 → テクスチャ付き3Dメッシュ' },
  'header.vramTier.tooltip': {
    en: 'GPU tier {0} — {1} MB VRAM',
    ja: 'GPUティア {0} — VRAM {1} MB',
  },
  'header.vramTier.unknown': {
    en: 'GPU tier unknown',
    ja: 'GPUティア不明',
  },

  // Stage labels
  'stage.extract_frames': { en: 'Extract Frames', ja: 'フレーム抽出' },
  'stage.colmap': { en: 'COLMAP', ja: 'COLMAP' },
  'stage.sam2': { en: 'SAM2', ja: 'SAM2' },
  'stage.milo': { en: 'MILo', ja: 'MILo' },
  'stage.texture': { en: 'Texture', ja: 'テクスチャ' },
  'stage.post_cleanup': { en: 'Post Cleanup', ja: '後処理' },

  // COLMAP config labels
  'config.colmap.gpu_sift': {
    en: 'GPU SIFT (requires CUDA COLMAP)',
    ja: 'GPU SIFT (CUDA版COLMAP必要)',
  },
  'config.colmap.dsp_sift': {
    en: 'DSP-SIFT (higher accuracy, slower)',
    ja: 'DSP-SIFT (高精度・低速)',
  },
  'config.colmap.first_octave': {
    en: 'First Octave -1 (2x upsample, more memory)',
    ja: 'First Octave -1 (2xアップサンプル・メモリ増)',
  },

  // Section titles
  'config.title': { en: 'Configuration', ja: '設定' },
  'log.title': { en: 'Log', ja: 'ログ' },
  'checkpoint.title': { en: 'Checkpoint', ja: 'チェックポイント' },

  // Overview
  'overview.title': { en: 'Objects', ja: 'オブジェクト一覧' },
  'overview.new_pipeline': { en: 'New Pipeline', ja: '新規パイプライン' },
  'overview.empty': { en: 'No objects yet. Start a new pipeline to create one.', ja: 'オブジェクトがありません。新規パイプラインを開始してください。' },
  'overview.open': { en: 'Open', ja: '開く' },
  'overview.stages': { en: 'Stages', ja: 'ステージ' },
  'overview.updated': { en: 'Updated', ja: '更新日時' },
  'overview.artifacts': { en: 'Artifacts', ja: '生成物' },

  // Breadcrumb
  'header.back_to_overview': { en: 'Pipeline', ja: 'パイプライン' },

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
  'settings.autoAccept': { en: 'Pipeline Mode', ja: 'パイプラインモード' },
  'settings.autoAccept.desc': {
    en: 'Auto-accept (skip confirmations except SAM2)',
    ja: '自動承認（SAM2以外の確認をスキップ）',
  },
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
