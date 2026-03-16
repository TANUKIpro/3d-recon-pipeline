/**
 * Pure functions for querying pipeline status objects.
 */

import { STAGE_COUNT, TRANSITION_STAGE_MAX } from './constants.js';

export function getStageInfo(statusMsg, stage) {
  return statusMsg?.stages?.[String(stage)] || null;
}

export function isStageDone(statusMsg, stage) {
  const info = getStageInfo(statusMsg, stage);
  if (!info) return false;
  if (info.status === 'complete') return true;
  return Number(info.progress) >= 100;
}

export function isStageAvailable(statusMsg, stage) {
  const info = getStageInfo(statusMsg, stage);
  if (!info) return false;
  return info.status === 'complete' || info.status === 'interactive' || Number(info.progress) > 0;
}

export function resolvePreferredStage(statusMsg) {
  if (!statusMsg) return 1;
  const current = Number(statusMsg.current_stage);
  if (Number.isFinite(current) && current >= 1 && current <= STAGE_COUNT) return current;
  for (let stage = STAGE_COUNT; stage >= 1; stage--) {
    if (isStageDone(statusMsg, stage)) return stage;
  }
  return 1;
}

export function buildStatusKey(statusMsg) {
  const objectName = statusMsg?.object_name || '';
  const next = statusMsg?.next_stage_confirmation || {};
  const cleanupProposal = statusMsg?.cleanup_proposal_path || '';
  const stageState = [];
  for (let i = 1; i <= STAGE_COUNT; i++) {
    const info = getStageInfo(statusMsg, i);
    stageState.push(`${info?.status || 'pending'}:${Math.round(Number(info?.progress) || 0)}`);
  }
  return [
    objectName,
    stageState.join('|'),
    next.required ? 'wait' : 'idle',
    next.from_stage || '',
    next.to_stage || '',
    cleanupProposal,
  ].join('|');
}

export function isTransitionConfirmed(statusMsg, stage) {
  if (!statusMsg || stage < 1 || stage > TRANSITION_STAGE_MAX) return false;
  if (!isStageDone(statusMsg, stage)) return false;

  const current = Number(statusMsg.current_stage);
  if (Number.isFinite(current) && current > stage) {
    return true;
  }

  const nextInfo = getStageInfo(statusMsg, stage + 1);
  if (!nextInfo) return false;
  if (nextInfo.status !== 'pending') return true;
  return Number(nextInfo.progress) > 0;
}

export function resolveTransitionTarget(stage, toStage = null) {
  const parsed = Number(toStage);
  if (Number.isFinite(parsed) && parsed >= 1 && parsed <= STAGE_COUNT) {
    return parsed;
  }
  return Math.max(1, Math.min(STAGE_COUNT, Number(stage) + 1));
}

export function defaultTaskConfirmIdleMessage(stage) {
  const target = resolveTransitionTarget(stage);
  return `Stage ${stage} confirmation will appear after completion (next: Stage ${target}).`;
}

export function defaultTaskConfirmWaitingMessage(stage, toStage = null) {
  const target = resolveTransitionTarget(stage, toStage);
  if (target === Number(stage)) {
    return `Stage ${stage} step complete. Confirm to continue within Stage ${stage}.`;
  }
  return `Stage ${stage} complete. Confirm to continue to Stage ${target}.`;
}

export function defaultTaskConfirmConfirmedMessage(stage, toStage = null) {
  const target = resolveTransitionTarget(stage, toStage);
  if (target === Number(stage)) {
    return `Stage ${stage} step confirmed. Continuing Stage ${stage} workflow.`;
  }
  return `Stage ${stage} confirmed. Proceeded to Stage ${target}.`;
}

export function defaultTaskConfirmStandbyMessage(stage, toStage = null) {
  const target = resolveTransitionTarget(stage, toStage);
  if (target === Number(stage)) {
    return `Stage ${stage} is complete. Start the pipeline to continue remaining Stage ${stage} tasks.`;
  }
  return `Stage ${stage} is complete. Start the pipeline to continue to Stage ${target}.`;
}
