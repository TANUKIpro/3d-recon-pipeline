/**
 * Stage panel switching coordinator.
 *
 * Each stage pill in the stage bar acts as a clickable tab,
 * showing the corresponding stage-specific viewport panel.
 */

export class StageController {
  constructor() {
    this._stageCount = 5;
    this._panels = {};
    this._pills = {};
    this._activeStage = 1; // Default to Extract Frames panel
    this._stageStates = {};

    // Collect all pills and panels
    for (let i = 1; i <= this._stageCount; i++) {
      this._pills[i] = document.querySelector(`.stage-pill[data-stage="${i}"]`);
      this._panels[i] = document.getElementById(`stage-panel-${i}`);
      this._stageStates[i] = 'pending';
    }

    this._bindEvents();

    // Mark default panel
    this.activateStage(this._activeStage);
  }

  get activeStage() {
    return this._activeStage;
  }

  activateStage(n) {
    // Hide all panels, deselect all pills
    for (let i = 1; i <= this._stageCount; i++) {
      this._panels[i]?.classList.remove('active');
      this._pills[i]?.classList.remove('selected');
    }

    // Show target panel, select pill
    this._panels[n]?.classList.add('active');
    this._pills[n]?.classList.add('selected');
    this._activeStage = n;

    // Dispatch custom event for lazy 3D init
    document.dispatchEvent(new CustomEvent('stage-activated', { detail: { stage: n } }));
  }

  setStageState(n, state) {
    this._stageStates[n] = state;
  }

  getStageState(n) {
    return this._stageStates[n] || 'pending';
  }

  _bindEvents() {
    for (let i = 1; i <= this._stageCount; i++) {
      const pill = this._pills[i];
      if (pill) {
        pill.addEventListener('click', () => this.activateStage(i));
      }
    }
  }
}
