import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';

import { PostTextureCleanupReviewController } from '../../../scripts/dashboard/static/js/post-texture-cleanup-review.js';

function buildDom() {
  document.body.innerHTML = `
    <div id="post-texture-cleanup-review" class="hidden"></div>
    <div id="post-cleanup-review-summary"></div>
    <div id="post-cleanup-review-stats"></div>
    <button id="post-cleanup-apply">Apply</button>
    <button id="post-cleanup-skip">Skip</button>
    <input id="post-cleanup-overlay-toggle" type="checkbox" checked>
  `;
}

describe('PostTextureCleanupReviewController', () => {
  let preview;
  let stageCtrl;
  let appendLog;

  beforeEach(() => {
    buildDom();
    document.getElementById('post-cleanup-overlay-toggle').checked = true;
    preview = {
      loadStageResult: vi.fn().mockResolvedValue(undefined),
      loadStageOverlay: vi.fn().mockResolvedValue(undefined),
      clearStageOverlay: vi.fn(),
    };
    stageCtrl = { activateStage: vi.fn() };
    appendLog = vi.fn();
    global.fetch = vi.fn();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    delete global.fetch;
  });

  it('showReview renders proposal and loads the red overlay', async () => {
    const ctrl = new PostTextureCleanupReviewController({ preview, stageCtrl, appendLog });
    const proposal = {
      recommended_decision: 'apply',
      artifacts: { removed_region_ply: 'post_texture_contact_cleanup/proposal_removed_region.ply' },
      summary: { removed_face_count: 12, cap_face_count: 4, confidence: 0.87 },
      ground_plane: { selected_shift: 0.012345, selected_loop_area: 0.456789 },
      mask_consistency: { visible_samples: 42, object_outside_ratio: 0.25, ground_overlap_ratio: 0.5 },
    };

    await ctrl.showReview(proposal);
    await ctrl._syncOverlay();

    expect(preview.loadStageResult).toHaveBeenCalledWith(6, { file: 'p5_texture/textured_mesh.obj' });
    expect(preview.loadStageOverlay).toHaveBeenCalledWith(
      6,
      'p6_cleanup/post_texture_contact_cleanup/proposal_removed_region.ply',
      { color: 0xff5533, opacity: 0.68 },
    );
    expect(stageCtrl.activateStage).toHaveBeenCalledWith(6);
    expect(document.getElementById('post-texture-cleanup-review').classList.contains('hidden')).toBe(false);
    expect(document.getElementById('post-cleanup-review-summary').textContent).toContain('removed faces: 12');
  });

  it('syncFromStatus fetches the proposal when stage 6 is interactive', async () => {
    const ctrl = new PostTextureCleanupReviewController({ preview, stageCtrl, appendLog });
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({
          proposal: {
            recommended_decision: 'skip',
            artifacts: {},
          stats: {},
        },
      }),
    });

    await ctrl.syncFromStatus({
      current_stage: 6,
      running: true,
      cleanup_proposal_path: '/tmp/proposal.json',
      stages: { '6': { status: 'interactive' } },
    });

    expect(global.fetch).toHaveBeenCalledWith('/api/post-texture-cleanup/proposal');
    expect(stageCtrl.activateStage).toHaveBeenCalledWith(6);
  });

  it('confirm posts the selected decision and appends a log line', async () => {
    const ctrl = new PostTextureCleanupReviewController({ preview, stageCtrl, appendLog });
    global.fetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          proposal: { recommended_decision: 'apply', artifacts: {}, stats: {} },
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'confirmed', decision: 'apply' }),
      });

    await ctrl.showReview({ recommended_decision: 'apply', artifacts: {}, stats: {} });
    await ctrl.confirm('apply');

    expect(global.fetch).toHaveBeenLastCalledWith('/api/post-texture-cleanup/confirm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision: 'apply' }),
    });
    expect(appendLog).toHaveBeenCalledWith('stdout', 'Post-texture cleanup decision: apply\n', { stage: 6 });
  });
});
