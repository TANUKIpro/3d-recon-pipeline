import { describe, it, expect } from 'vitest';
import {
  getStageInfo,
  isStageDone,
  isStageAvailable,
  resolvePreferredStage,
  buildStatusKey,
  isTransitionConfirmed,
  resolveTransitionTarget,
  defaultTaskConfirmIdleMessage,
  defaultTaskConfirmWaitingMessage,
  defaultTaskConfirmConfirmedMessage,
  defaultTaskConfirmStandbyMessage,
} from '../../../scripts/dashboard/static/js/pipeline-status.js';

// ── Helper to build a status message ──────────────────────────

function mkStatus(overrides = {}) {
  return {
    object_name: 'test-obj',
    running: false,
    current_stage: null,
    next_stage_confirmation: {},
    stages: {},
    ...overrides,
  };
}

function mkStage(status = 'pending', progress = 0) {
  return { status, progress };
}

// ── getStageInfo ──────────────────────────────────────────────

describe('getStageInfo', () => {
  it('returns null for missing status', () => {
    expect(getStageInfo(null, 1)).toBe(null);
  });

  it('returns null for missing stage', () => {
    expect(getStageInfo(mkStatus(), 1)).toBe(null);
  });

  it('returns stage info when present', () => {
    const status = mkStatus({ stages: { '1': mkStage('running', 50) } });
    expect(getStageInfo(status, 1)).toEqual({ status: 'running', progress: 50 });
  });
});

// ── isStageDone ───────────────────────────────────────────────

describe('isStageDone', () => {
  it('returns false for null status', () => {
    expect(isStageDone(null, 1)).toBe(false);
  });

  it('returns false for missing stage', () => {
    expect(isStageDone(mkStatus(), 1)).toBe(false);
  });

  it('returns true for complete status', () => {
    const status = mkStatus({ stages: { '1': mkStage('complete', 100) } });
    expect(isStageDone(status, 1)).toBe(true);
  });

  it('returns true for progress >= 100', () => {
    const status = mkStatus({ stages: { '1': mkStage('running', 100) } });
    expect(isStageDone(status, 1)).toBe(true);
  });

  it('returns false for progress < 100', () => {
    const status = mkStatus({ stages: { '1': mkStage('running', 99) } });
    expect(isStageDone(status, 1)).toBe(false);
  });
});

// ── isStageAvailable ──────────────────────────────────────────

describe('isStageAvailable', () => {
  it('returns false for null status', () => {
    expect(isStageAvailable(null, 1)).toBe(false);
  });

  it('returns true for complete stage', () => {
    const status = mkStatus({ stages: { '1': mkStage('complete') } });
    expect(isStageAvailable(status, 1)).toBe(true);
  });

  it('returns true for interactive stage', () => {
    const status = mkStatus({ stages: { '3': mkStage('interactive') } });
    expect(isStageAvailable(status, 3)).toBe(true);
  });

  it('returns true for stage with progress > 0', () => {
    const status = mkStatus({ stages: { '2': mkStage('running', 10) } });
    expect(isStageAvailable(status, 2)).toBe(true);
  });

  it('returns false for pending stage with no progress', () => {
    const status = mkStatus({ stages: { '2': mkStage('pending', 0) } });
    expect(isStageAvailable(status, 2)).toBe(false);
  });
});

// ── resolvePreferredStage ─────────────────────────────────────

describe('resolvePreferredStage', () => {
  it('returns 1 for null status', () => {
    expect(resolvePreferredStage(null)).toBe(1);
  });

  it('returns current_stage when valid', () => {
    const status = mkStatus({ current_stage: 3 });
    expect(resolvePreferredStage(status)).toBe(3);
  });

  it('returns highest done stage when no current_stage', () => {
    const status = mkStatus({
      stages: {
        '1': mkStage('complete', 100),
        '2': mkStage('complete', 100),
        '3': mkStage('running', 50),
      },
    });
    expect(resolvePreferredStage(status)).toBe(2);
  });

  it('returns 1 when no stages are done', () => {
    const status = mkStatus({ stages: { '1': mkStage('running', 30) } });
    expect(resolvePreferredStage(status)).toBe(1);
  });
});

// ── buildStatusKey ────────────────────────────────────────────

describe('buildStatusKey', () => {
  it('produces a deterministic key', () => {
    const status = mkStatus({
      object_name: 'obj',
      stages: { '1': mkStage('complete', 100) },
    });
    const key1 = buildStatusKey(status);
    const key2 = buildStatusKey(status);
    expect(key1).toBe(key2);
  });

  it('includes object name', () => {
    const status = mkStatus({ object_name: 'my-object' });
    expect(buildStatusKey(status)).toContain('my-object');
  });

  it('includes wait state when confirmation required', () => {
    const status = mkStatus({
      next_stage_confirmation: { required: true, from_stage: 2, to_stage: 3 },
    });
    expect(buildStatusKey(status)).toContain('wait');
  });

  it('includes idle state when no confirmation required', () => {
    const status = mkStatus();
    expect(buildStatusKey(status)).toContain('idle');
  });
});

// ── isTransitionConfirmed ─────────────────────────────────────

describe('isTransitionConfirmed', () => {
  it('returns false for null status', () => {
    expect(isTransitionConfirmed(null, 1)).toBe(false);
  });

  it('returns false for stage out of range', () => {
    expect(isTransitionConfirmed(mkStatus(), 0)).toBe(false);
    expect(isTransitionConfirmed(mkStatus(), 8)).toBe(false);
  });

  it('returns false when stage is not done', () => {
    const status = mkStatus({ stages: { '1': mkStage('running', 50) } });
    expect(isTransitionConfirmed(status, 1)).toBe(false);
  });

  it('returns true when current_stage is higher', () => {
    const status = mkStatus({
      current_stage: 3,
      stages: { '1': mkStage('complete', 100) },
    });
    expect(isTransitionConfirmed(status, 1)).toBe(true);
  });

  it('returns true when next stage is not pending', () => {
    const status = mkStatus({
      stages: {
        '1': mkStage('complete', 100),
        '2': mkStage('running', 10),
      },
    });
    expect(isTransitionConfirmed(status, 1)).toBe(true);
  });

  it('returns false when next stage is pending with no progress', () => {
    const status = mkStatus({
      stages: {
        '1': mkStage('complete', 100),
        '2': mkStage('pending', 0),
      },
    });
    expect(isTransitionConfirmed(status, 1)).toBe(false);
  });
});

// ── resolveTransitionTarget ───────────────────────────────────

describe('resolveTransitionTarget', () => {
  it('returns toStage when valid', () => {
    expect(resolveTransitionTarget(3, 5)).toBe(5);
  });

  it('returns stage+1 when toStage is null', () => {
    expect(resolveTransitionTarget(3, null)).toBe(4);
  });

  it('returns stage+1 when toStage is invalid', () => {
    expect(resolveTransitionTarget(3, 'abc')).toBe(4);
  });

  it('clamps to STAGE_COUNT', () => {
    expect(resolveTransitionTarget(8, null)).toBe(8);
  });

  it('clamps to 1 minimum', () => {
    expect(resolveTransitionTarget(0, null)).toBe(1);
  });
});

// ── defaultTaskConfirmIdleMessage ─────────────────────────────

describe('defaultTaskConfirmIdleMessage', () => {
  it('includes stage number and next stage', () => {
    const msg = defaultTaskConfirmIdleMessage(3);
    expect(msg).toContain('Stage 3');
    expect(msg).toContain('Stage 4');
  });
});

// ── defaultTaskConfirmWaitingMessage ──────────────────────────

describe('defaultTaskConfirmWaitingMessage', () => {
  it('says complete + next stage for cross-stage transition', () => {
    const msg = defaultTaskConfirmWaitingMessage(3, 4);
    expect(msg).toContain('Stage 3 complete');
    expect(msg).toContain('Stage 4');
  });

  it('says step complete for same-stage transition', () => {
    const msg = defaultTaskConfirmWaitingMessage(3, 3);
    expect(msg).toContain('step complete');
    expect(msg).toContain('within Stage 3');
  });
});

// ── defaultTaskConfirmConfirmedMessage ─────────────────────────

describe('defaultTaskConfirmConfirmedMessage', () => {
  it('says confirmed + proceeded for cross-stage', () => {
    const msg = defaultTaskConfirmConfirmedMessage(3, 4);
    expect(msg).toContain('Stage 3 confirmed');
    expect(msg).toContain('Stage 4');
  });

  it('says step confirmed for same-stage', () => {
    const msg = defaultTaskConfirmConfirmedMessage(3, 3);
    expect(msg).toContain('step confirmed');
  });
});

// ── defaultTaskConfirmStandbyMessage ──────────────────────────

describe('defaultTaskConfirmStandbyMessage', () => {
  it('says complete + start pipeline for cross-stage', () => {
    const msg = defaultTaskConfirmStandbyMessage(3, 4);
    expect(msg).toContain('Stage 3 is complete');
    expect(msg).toContain('Stage 4');
  });

  it('says remaining tasks for same-stage', () => {
    const msg = defaultTaskConfirmStandbyMessage(3, 3);
    expect(msg).toContain('remaining Stage 3 tasks');
  });
});
