/**
 * Settings modal panel — theme, language, log preferences.
 * All settings persisted to localStorage.
 */

const KEYS = {
  theme: 'im2pc:theme',
  lang: 'im2pc:lang',
  autoScroll: 'im2pc:log.autoScroll',
  maxLines: 'im2pc:log.maxLines',
};

const DEFAULTS = {
  theme: 'dark',
  lang: 'en',
  autoScroll: true,
  maxLines: 2000,
};

export class SettingsPanel {
  /**
   * @param {import('./i18n.js').I18n} i18n
   */
  constructor(i18n) {
    this._i18n = i18n;

    // Read stored values (apply theme/lang immediately, before first paint)
    this._theme = localStorage.getItem(KEYS.theme) || DEFAULTS.theme;
    this._lang = localStorage.getItem(KEYS.lang) || i18n.lang || DEFAULTS.lang;
    this._autoScroll = this._readBool(KEYS.autoScroll, DEFAULTS.autoScroll);
    this._maxLines = this._readInt(KEYS.maxLines, DEFAULTS.maxLines, 100, 50000);

    // Callbacks
    this.onThemeChanged = null;   // (theme: string) => void
    this.onLangChanged = null;    // (lang: string) => void
    this.onLogSettingsChanged = null; // ({autoScroll, maxLines}) => void

    // Apply theme immediately
    this._applyTheme(this._theme);

    // Sync i18n lang
    if (this._lang !== i18n.lang) {
      i18n.setLang(this._lang);
    }

    // Build DOM
    this._overlay = null;
    this._buildDOM();

    // Bind gear button
    const gearBtn = document.getElementById('settings-btn');
    if (gearBtn) {
      gearBtn.addEventListener('click', () => this.toggle());
    }
  }

  get theme() { return this._theme; }
  get lang() { return this._lang; }
  get autoScroll() { return this._autoScroll; }
  get maxLines() { return this._maxLines; }

  open() {
    if (this._overlay) {
      this._overlay.classList.add('open');
    }
  }

  close() {
    if (this._overlay) {
      this._overlay.classList.remove('open');
    }
  }

  toggle() {
    if (this._overlay?.classList.contains('open')) {
      this.close();
    } else {
      this.open();
    }
  }

  _readBool(key, fallback) {
    const val = localStorage.getItem(key);
    if (val === null) return fallback;
    return val !== 'false';
  }

  _readInt(key, fallback, min, max) {
    const raw = localStorage.getItem(key);
    if (raw === null) return fallback;
    const parsed = parseInt(raw, 10);
    if (!Number.isFinite(parsed)) return fallback;
    return Math.max(min, Math.min(max, parsed));
  }

  _applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
  }

  _buildDOM() {
    const t = (key) => this._i18n.t(key);

    // Overlay
    this._overlay = document.createElement('div');
    this._overlay.className = 'modal-overlay';
    this._overlay.addEventListener('click', (e) => {
      if (e.target === this._overlay) this.close();
    });

    // Modal content
    const modal = document.createElement('div');
    modal.className = 'modal-content';

    // Header
    const header = document.createElement('div');
    header.className = 'modal-header';
    const title = document.createElement('h3');
    title.textContent = t('settings.title');
    title.setAttribute('data-i18n', 'settings.title');
    const closeBtn = document.createElement('button');
    closeBtn.className = 'modal-close-btn';
    closeBtn.textContent = '\u2715';
    closeBtn.title = t('settings.close');
    closeBtn.addEventListener('click', () => this.close());
    header.appendChild(title);
    header.appendChild(closeBtn);
    modal.appendChild(header);

    // Body
    const body = document.createElement('div');
    body.className = 'modal-body';

    // Theme
    body.appendChild(this._buildThemeGroup(t));

    // Language
    body.appendChild(this._buildLangGroup(t));

    // Log settings
    body.appendChild(this._buildLogGroup(t));

    modal.appendChild(body);
    this._overlay.appendChild(modal);
    document.body.appendChild(this._overlay);

    // Escape key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this._overlay.classList.contains('open')) {
        this.close();
      }
    });
  }

  _buildThemeGroup(t) {
    const group = document.createElement('div');
    group.className = 'settings-group';

    const label = document.createElement('label');
    label.className = 'settings-label';
    label.textContent = t('settings.theme');
    label.setAttribute('data-i18n', 'settings.theme');
    group.appendChild(label);

    const row = document.createElement('div');
    row.className = 'settings-row settings-toggle-row';

    const darkBtn = document.createElement('button');
    darkBtn.className = `settings-toggle-btn${this._theme === 'dark' ? ' active' : ''}`;
    darkBtn.textContent = t('settings.theme.dark');
    darkBtn.setAttribute('data-i18n', 'settings.theme.dark');
    darkBtn.dataset.value = 'dark';

    const lightBtn = document.createElement('button');
    lightBtn.className = `settings-toggle-btn${this._theme === 'light' ? ' active' : ''}`;
    lightBtn.textContent = t('settings.theme.light');
    lightBtn.setAttribute('data-i18n', 'settings.theme.light');
    lightBtn.dataset.value = 'light';

    const setTheme = (theme) => {
      this._theme = theme;
      localStorage.setItem(KEYS.theme, theme);
      this._applyTheme(theme);
      darkBtn.classList.toggle('active', theme === 'dark');
      lightBtn.classList.toggle('active', theme === 'light');
      this.onThemeChanged?.(theme);
    };

    darkBtn.addEventListener('click', () => setTheme('dark'));
    lightBtn.addEventListener('click', () => setTheme('light'));

    row.appendChild(darkBtn);
    row.appendChild(lightBtn);
    group.appendChild(row);
    return group;
  }

  _buildLangGroup(t) {
    const group = document.createElement('div');
    group.className = 'settings-group';

    const label = document.createElement('label');
    label.className = 'settings-label';
    label.textContent = t('settings.language');
    label.setAttribute('data-i18n', 'settings.language');
    group.appendChild(label);

    const row = document.createElement('div');
    row.className = 'settings-row settings-toggle-row';

    const enBtn = document.createElement('button');
    enBtn.className = `settings-toggle-btn${this._lang === 'en' ? ' active' : ''}`;
    enBtn.textContent = 'English';
    enBtn.dataset.value = 'en';

    const jaBtn = document.createElement('button');
    jaBtn.className = `settings-toggle-btn${this._lang === 'ja' ? ' active' : ''}`;
    jaBtn.textContent = '\u65E5\u672C\u8A9E';
    jaBtn.dataset.value = 'ja';

    const setLang = (lang) => {
      this._lang = lang;
      localStorage.setItem(KEYS.lang, lang);
      this._i18n.setLang(lang);
      enBtn.classList.toggle('active', lang === 'en');
      jaBtn.classList.toggle('active', lang === 'ja');
      // Re-translate modal's own labels
      this._overlay.querySelectorAll('[data-i18n]').forEach((el) => {
        el.textContent = this._i18n.t(el.getAttribute('data-i18n'));
      });
      this.onLangChanged?.(lang);
    };

    enBtn.addEventListener('click', () => setLang('en'));
    jaBtn.addEventListener('click', () => setLang('ja'));

    row.appendChild(enBtn);
    row.appendChild(jaBtn);
    group.appendChild(row);
    return group;
  }

  _buildLogGroup(t) {
    const group = document.createElement('div');
    group.className = 'settings-group';

    const label = document.createElement('label');
    label.className = 'settings-label';
    label.textContent = t('settings.log');
    label.setAttribute('data-i18n', 'settings.log');
    group.appendChild(label);

    // Auto-scroll toggle
    const scrollRow = document.createElement('div');
    scrollRow.className = 'settings-row';

    const scrollLabel = document.createElement('label');
    scrollLabel.className = 'settings-check-label';
    const scrollCheck = document.createElement('input');
    scrollCheck.type = 'checkbox';
    scrollCheck.checked = this._autoScroll;
    scrollCheck.addEventListener('change', () => {
      this._autoScroll = scrollCheck.checked;
      localStorage.setItem(KEYS.autoScroll, String(this._autoScroll));
      this.onLogSettingsChanged?.({
        autoScroll: this._autoScroll,
        maxLines: this._maxLines,
      });
    });
    const scrollText = document.createElement('span');
    scrollText.textContent = t('settings.log.autoscroll');
    scrollText.setAttribute('data-i18n', 'settings.log.autoscroll');
    scrollLabel.appendChild(scrollCheck);
    scrollLabel.appendChild(scrollText);
    scrollRow.appendChild(scrollLabel);
    group.appendChild(scrollRow);

    // Max lines
    const linesRow = document.createElement('div');
    linesRow.className = 'settings-row';

    const linesLabel = document.createElement('label');
    linesLabel.className = 'settings-input-label';
    const linesText = document.createElement('span');
    linesText.textContent = t('settings.log.maxlines');
    linesText.setAttribute('data-i18n', 'settings.log.maxlines');
    const linesInput = document.createElement('input');
    linesInput.type = 'number';
    linesInput.value = String(this._maxLines);
    linesInput.min = '100';
    linesInput.max = '50000';
    linesInput.step = '100';
    linesInput.className = 'settings-number-input';
    linesInput.addEventListener('change', () => {
      const val = parseInt(linesInput.value, 10);
      if (Number.isFinite(val) && val >= 100 && val <= 50000) {
        this._maxLines = val;
        localStorage.setItem(KEYS.maxLines, String(val));
        this.onLogSettingsChanged?.({
          autoScroll: this._autoScroll,
          maxLines: this._maxLines,
        });
      }
    });
    linesLabel.appendChild(linesText);
    linesLabel.appendChild(linesInput);
    linesRow.appendChild(linesLabel);
    group.appendChild(linesRow);

    return group;
  }
}
