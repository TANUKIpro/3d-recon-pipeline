# clip2mesh リファクタリング計画

> **目的**: コードの保守性・可読性・拡張性を向上させる。動作を変えずに構造を改善する。
>
> **原則**:
> - 各ステップは独立してコミット可能な単位とする
> - 既存テスト (pytest + vitest) が全てパスすることを各ステップ完了の条件とする
> - ハルシネーション防止: 各ステップの「事前確認」で現在のコード状態を必ず読み取ってから作業する

---

## Phase 0: 事前準備（リファクタリング前の基盤整備） ✅

### 0-1. テストの実行確認
- [x] `npm test` でフロントエンドテスト全パスを確認
- [x] `docker compose -f docker-compose.test.yml run --rm test` で Python テスト全パスを確認（GPU 不要テストのみ）
- [x] 失敗テストがあれば先に修正する（リファクタリングとは別コミット）

### 0-2. リファクタリング用ブランチ作成
- [x] `git checkout -b refactor/phase-1` でブランチを作成
- [ ] 各フェーズ完了時に main へ PR & マージ

---

## Phase 1: Python パッケージ構造の整備 ✅

**目標**: `sys.path.insert` ハックを排除し、正規の Python パッケージインポートに統一する

コミット: `05afb8f refactor: replace sys.path hacks with proper package imports`

### 1-1. `__init__.py` の追加
- [x] `scripts/__init__.py` を空ファイルとして作成
- [x] `scripts/dashboard/__init__.py` は既存のため変更不要
- [x] `pyproject.toml` に `pythonpath = ["."]` を追加（ローカル pytest 用）

### 1-2. `sys.path.insert` の排除
- [x] `scripts/pipeline.py` — `import sys` + `sys.path.insert(...)` を削除
- [x] `scripts/dashboard/pipeline_runner.py` — `import sys` + `_SCRIPTS_DIR` 定義 + `sys.path.insert(...)` を削除
- [x] Dockerfile の `PYTHONPATH` を `/app/scripts` → `/app` に変更

### 1-3. bare name import の修正
- [x] `scripts/*.py` 内の bare name import を `scripts.` prefix 付きに変更（12 ファイル 18 箇所）
- [x] `scripts/dashboard/*.py` 内の bare name import を `scripts.` prefix 付きに変更（3 ファイル 24 箇所）
- [x] `scripts/pipeline.py` 内の遅延 import 10 箇所を修正

### 1-4. テスト側 import の整合
- [x] テストファイルの通常 import は既に `scripts.` prefix 使用済みのため変更不要
- [x] `sys.modules` パッチのキーを bare name → `scripts.` prefix に修正（3 ファイル）
  - `tests/dashboard/backend/test_sam2_service.py`
  - `tests/dashboard/backend/test_stage_wrappers.py`
  - `tests/dashboard/backend/test_app_endpoints.py`
- [x] Docker テスト全パス確認（469 passed）

---

## Phase 2: `object_store.py` の公開 API 命名修正 ✅

**目標**: `_` prefix の関数が外部から import されている矛盾を解消する

コミット: `3c1e5c0 refactor: remove _ prefix from 14 public functions in object_store.py`

### 2-1. 公開関数の `_` prefix 除去

**対象関数（14 関数）:**
- [x] `_infer_resume_stage` → `infer_resume_stage`
- [x] `_list_objects` → `list_objects`
- [x] `_list_preview_files` → `list_preview_files`
- [x] `_object_dir` → `object_dir`
- [x] `_prepare_object_output_dir` → `prepare_object_output_dir`
- [x] `_reset_outputs_from_stage` → `reset_outputs_from_stage`
- [x] `_resolve_output_root` → `resolve_output_root`
- [x] `_safe_json_load` → `safe_json_load`
- [x] `_sanitize_object_name` → `sanitize_object_name`
- [x] `_suggest_object_name` → `suggest_object_name`
- [x] `_summarize_object` → `summarize_object`
- [x] `_validate_object_name` → `validate_object_name`
- [x] `_validate_resume_prerequisites` → `validate_resume_prerequisites`
- [x] `_write_object_meta` → `write_object_meta`

- [x] `object_store.py`, `app.py`, `test_object_store.py`, `test_app_endpoints.py` の全参照箇所を一括置換
- [x] 内部専用ヘルパー（`_utc_iso`, `_stage_completion_flags`, `_latest_update_ts`, `_objects_root`）は `_` のまま維持
- [x] Docker で全テスト実行して確認（469 passed）

---

## Phase 3: 巨大ファイルの分割 ✅

**目標**: 800 行超のファイルを論理的な単位に分割し、各ファイルの責務を明確にする

**戦略**: 関数を新しいサブモジュールに移動し、元ファイルは全名前を re-export する薄いシムとして残す。
これにより既存の import パスは一切変更不要。

### 3-1. `stage_texture_bake.py`（2910 行）→ `scripts/texture/` ✅

コミット: `076d560 refactor: split stage_texture_bake.py into scripts/texture/ subpackage`

| 新ファイル | 責務 |
|---|---|
| `scripts/texture/__init__.py` | `bake_texture` を re-export |
| `scripts/texture/progress.py` | 進捗・メモリ (`ProgressCallback`, `_emit_progress`, `_get_available_memory_mb`) |
| `scripts/texture/io_utils.py` | データ読込・キャッシュ (`_FrameCache`, `_load_point_cloud`, `_load_poses` 等) |
| `scripts/texture/intrinsics.py` | カメラ内部パラメータ推定 (`_make_K`, `_project_points`, `_estimate_intrinsics`) |
| `scripts/texture/config.py` | テクスチャ設定解決 (`_resolve_texture_device`, `_resolve_texture_size` 等) |
| `scripts/texture/image_utils.py` | 画像処理ユーティリティ (`_bilinear_sample`, `_rgb_to_gray` 等) |
| `scripts/texture/view_scoring.py` | ビュースコアリング (`_rasterize_view_depth`, `_evaluate_view_samples` 等) |
| `scripts/texture/conflict_region.py` | コンフリクト検出・領域分析 + 定数群 |
| `scripts/texture/seam_blend.py` | シームブレンディング + 定数群 |
| `scripts/texture/cap_region.py` | キャップ領域処理 |
| `scripts/texture/bake.py` | メインオーケストレータ (`bake_texture()`) |

- [x] `stage_texture_bake.py` を re-export シムに変換
- [x] テスト7箇所の mock パッチパスを `scripts.texture.config.torch` に更新
- [x] 全469テストパス確認

### 3-2. `stage_contact_hole_repair.py`（2189 行）→ `scripts/repair/` ✅

コミット: `00dfbc5 refactor: split stage_contact_hole_repair.py into scripts/repair/ subpackage`

| 新ファイル | 責務 |
|---|---|
| `scripts/repair/__init__.py` | 公開 API re-export |
| `scripts/repair/types.py` | 10 dataclasses + `ProgressCallback`, `CancelCallback`, `_emit_progress`, `_check_cancel` |
| `scripts/repair/config.py` | パラメータ解決 (`_env_bool`, `_env_float`, `_env_int`, `_resolve_params`) |
| `scripts/repair/mesh_io.py` | メッシュ I/O (`_write_mesh_safe`, `_clean_mesh` 等) |
| `scripts/repair/boundary.py` | 境界検出・ループ解析 (10 関数) |
| `scripts/repair/triangulate.py` | 三角形化 (5 関数) |
| `scripts/repair/candidates.py` | 候補収集・分析 |
| `scripts/repair/ground_plane.py` | 地面平面検出・クリッピング (18 関数) |
| `scripts/repair/pipeline.py` | 修復パイプライン (`run_contact_hole_repair` 等) |

- [x] `stage_contact_hole_repair.py` を re-export シムに変換（`__main__` ブロック保持）
- [x] 全469テストパス確認

### 3-3. `app.py`（976 行）→ `scripts/dashboard/routes/` ✅

コミット: `e5aa3f1 refactor: split app.py routes into scripts/dashboard/routes/ subpackage`

| 新ファイル | ルート群 |
|---|---|
| `scripts/dashboard/routes/__init__.py` | 空 |
| `scripts/dashboard/routes/pipeline.py` | `/api/pipeline/*` (10 ハンドラ) |
| `scripts/dashboard/routes/sam2.py` | `/api/sam2/*` (11 ハンドラ) |
| `scripts/dashboard/routes/verification.py` | `/api/verification/*` (2 ハンドラ) |
| `scripts/dashboard/routes/preview.py` | `/api/preview/*` (4 ハンドラ) |
| `scripts/dashboard/routes/mesh.py` | `/api/mesh-repair/*`, `/api/mesh/*` (3 ハンドラ) |
| `scripts/dashboard/routes/health.py` | `/api/vram` (1 ハンドラ) |

- [x] `APIRouter` で各ルートファイルに分離
- [x] 共有状態は `import scripts.dashboard.app as _app` + ハンドラ内 `_app.X` アクセスで循環 import 回避
- [x] `app.py` に31ハンドラ名を re-export（テスト後方互換）
- [x] 全469テストパス確認

### 3-4. `pipeline_runner.py`（1083 行）— スキップ

**理由**: 単一のオーケストレータとして凝集度が高い。SAM2 インタラクティブループとメッシュ修復ループは
`session` 状態と async Event を密に共有しており、分離するとパラメータ爆発を招く。

### 3-5. フロントエンド JS 分割 ✅

コミット: `7fe22a5 refactor: split preview.js and config-panel.js into submodules`

#### `preview.js`（1741 行 → ~780 行）→ `js/preview/`

| 新ファイル | 責務 |
|---|---|
| `js/preview/constants.js` | `FIRST_MESH_PREVIEW_FILES`, `SCENE_THEMES` |
| `js/preview/pose-utils.js` | カメラポーズ・シーンフリップ (13 mixin methods) |
| `js/preview/mesh-repair-overlay.js` | メッシュ修復オーバーレイ (14 mixin methods) |
| `js/preview/scene-helpers.js` | シーンユーティリティ (16 mixin methods) |

#### `config-panel.js`（1566 行 → ~1100 行）→ `js/config/`

| 新ファイル | 責務 |
|---|---|
| `js/config/presets.js` | プリセット定数 (denoise, classical, meshwrap, mesh repair) |
| `js/config/frame-budget.js` | Pi3X フレーム予算計算 (12 mixin methods) |
| `js/config/form-helpers.js` | フォームユーティリティ (17 mixin methods) |

- [x] `Object.assign(Class.prototype, {...})` による mixin パターンで移行
- [x] THREE.js 遅延ロードは `initThree(module)` export で対応
- [x] 全554フロントエンドテストパス確認

---

## Phase 4: `app.py` の共有状態管理の改善 ✅

**目標**: グローバル変数によるモジュール間結合を緩和する

**設計判断**: テストが全ハンドラを直接コルーチン呼び出ししているため（TestClient 不使用）、
`Depends()` をハンドラシグネチャに追加するとテスト全壊する。代わりに `get_state()`
アクセサ関数をハンドラ本体内で呼ぶパターンを採用。

### 4-1. 依存コンテナの導入

- [x] `scripts/dashboard/dependencies.py` を新規作成
  - `AppState` クラス（`session`, `sam2_service`, `input_dir`, `output_dir`）
  - `active_output_dir()`, `load_object_into_session()` メソッド（元 `app.py` のヘルパー）
  - 定数移動: `VIDEO_EXTENSIONS`, `MESH_POSTPROCESS_METHODS`
  - シングルトン管理: `init_state()`, `get_state()`
- [x] `app.py` を薄い composition root に変換
  - `init_state()` で `_app_state` 作成
  - 後方互換エイリアス維持: `session`, `sam2_service`, `INPUT_DIR`, `OUTPUT_DIR`
  - `_active_output_dir()`, `_load_object_into_session()` は薄いラッパーとして残存
- [x] 全5ルートファイルから `import scripts.dashboard.app as _app` を除去
  - `routes/verification.py`: `get_state().active_output_dir()`
  - `routes/sam2.py`: `get_state().session`, `get_state().sam2_service`
  - `routes/preview.py`: `get_state()` + `object_store` 直接 import
  - `routes/mesh.py`: `get_state()` + `MESH_POSTPROCESS_METHODS`, `STAGE_RESET_PATHS`, `broadcast`
  - `routes/pipeline.py`: `get_state()` + `object_store`, `pipeline_runner` 直接 import
- [x] テストのパッチパス更新（~25 箇所）
  - `_active_output_dir` → `AppState.active_output_dir`
  - `OUTPUT_DIR`/`INPUT_DIR` パッチ → `get_state().output_dir`/`input_dir` 直接設定
  - `broadcast`/`run_pipeline` 等 → ルートモジュールのパス
- [x] 全462 Pythonテストパス、全554フロントエンドテストパス確認

---

## Phase 5: `config_defaults.py` 定数命名の一貫性修正 ✅

**目標**: `_` prefix（internal）と prefix なし（public/UI-facing）の区別が実態と一致しない定数を修正する

コミット: `b770e13 refactor: remove _ prefix from 3 public constants in config_defaults.py`

### 5-1. `_MESHWRAP_METHOD` → `MESHWRAP_METHOD`
- [x] `config_defaults.py`: INTERNAL → USER-CONFIGURABLE Stage 6 セクションへ移動、`_` prefix 除去
- [x] 元の位置に後方互換エイリアス `_MESHWRAP_METHOD = MESHWRAP_METHOD` を残す
- [x] `dashboard/state.py`: import 名と `PipelineConfig.meshwrap_method` のデフォルト値を更新
- [x] `dashboard/configuration.py`: import 名と `parse_choice()` 呼び出しの参照を更新
- [x] `stage_mesh_wrap.py`: `MESHWRAP_METHOD as _DEFAULT_WRAP_METHOD` に変更

### 5-2. `_MESHWRAP_METHODS` → `MESHWRAP_METHODS`
- [x] `config_defaults.py`: `MESHWRAP_METHOD` と同じ位置へ移動、後方互換エイリアス追加
- [x] `dashboard/configuration.py`: import 名と `parse_choice()` 呼び出しの参照を更新
- [x] `stage_mesh_wrap.py`: `MESHWRAP_METHODS as _VALID_METHODS` に変更

### 5-3. `_OUTPUT_DIR_DEFAULT` → `OUTPUT_DIR_DEFAULT`
- [x] `config_defaults.py`: INTERNAL Infrastructure → USER-CONFIGURABLE セクション末尾へ移動、後方互換エイリアス追加
- [x] `dashboard/state.py`: import 名と `PipelineConfig.output_dir` のデフォルト値を更新

### スコープ外（実施しない）
- Import エイリアスの統一（4ファイル×44+箇所）: 各ファイル内で一貫しており、リスク/ベネフィット比が合わない
- `_REPAIR_MIN_LOOP_VERTICES` の削除: `repair/boundary.py` と `repair/pipeline.py` で使用中

- [x] 全462 Pythonテストパス確認（E2E 1件は Pi3X 依存で pre-existing failure）

---

## Phase 6: `stage_wrappers.py` の遅延 import パターン統一 ✅

**目標**: 各ラッパー関数内の遅延 import と呼び出しの間にある不要な空行を除去し、パターンを統一する

コミット: `refactor: unify lazy import pattern in stage_wrappers.py`

### 6-1. パターンの確認と統一

統一パターン: import 文の直後に空行なしで呼び出し（or `with` ブロック）。

- [x] 6 関数の import 後の不要な空行を除去:
  - `_stage_classical_mesh`
  - `_stage_mesh_wrap`
  - `_stage_mesh_repair`
  - `_stage_mesh_repair_analyze`
  - `_stage_mesh_repair_selected`
  - `_stage_extract_ground_plane`
- [x] 重複 import（`cleanup_pytorch_vram` が 2 関数で import）は遅延 import の正常な設計のため維持
- [x] Docker 全テスト実行して確認（469 passed）

---

## Phase 7: フロントエンド CSS の分割 ✅

**目標**: `style.css`（1703 行）を論理単位に分割

### 7-1. CSS の分割 ✅

`style.css` を `@import` アグリゲータに変換し、9 個のサブモジュールに分割:

| 新ファイル | 責務 |
|---|---|
| `static/css/base.css` | リセット、`:root` 変数、ボタン、スクロールバー、ライトテーマ変数 |
| `static/css/layout.css` | ヘッダー、バッジ、パンくず、ビュー切替、メインレイアウト |
| `static/css/overview.css` | オーバービューグリッド、カード、サムネイル、アーティファクト |
| `static/css/pipeline.css` | ステージバー、ステージピル、コネクタ、メッシュ分岐、ステージパネル |
| `static/css/sam2.css` | SAM2 コンテナ、キャンバス、ツールバー、検証ストリップ |
| `static/css/preview.css` | 3D コンテナ、Pi3X ツールバー、メッシュステータス、ギャラリー |
| `static/css/config.css` | 設定パネル、ステージフィルタリング、フォームグループ |
| `static/css/panels.css` | チェックポイントパネル、ログパネル |
| `static/css/modal.css` | 設定ギアボタン、モーダルオーバーレイ、設定フォーム |

- [x] `style.css` のセクション構造を確認
- [x] 分割方針を最終決定（9 ファイル）
- [x] `style.css` を `@import` アグリゲータに変換（`index.html` 変更不要）
- [x] レスポンシブ `@media` ルールを各コンポーネントファイルに分配
- [x] ライトテーマ要素ルールを各コンポーネントファイルに移動
- [x] フロントエンドテスト実行して確認

---

## Phase 8: テスト構造の改善 ✅

**目標**: テストヘルパーの重複を排除し、カバレッジを可視化する

### 8-1. Python テストの conftest 整理 ✅

- [x] `FIXTURE_DIR` 定数を `tests/conftest.py` に集約
- [x] 4 ファイルのローカル定義を `from tests.conftest import FIXTURE_DIR` に置換
- [x] `fixture_dir()` pytest fixture が `FIXTURE_DIR` を返すように更新

### 8-2. フロントエンドテストヘルパーの整理 ✅

- [x] `buildMeshRepairDOM()` を `dom-factory.js` に移動
- [x] `mesh-repair-controller.test.js` のローカル定義を import に置換
- [x] フロントエンドテスト実行して確認

---

## Phase 9: HTML の構造改善

**目標**: `index.html`（756 行）のインラインスクリプト・スタイルを排除

### 9-1. HTML のクリーンアップ
> **事前確認**: `index.html` を読み、インラインの `<script>` や `<style>` を確認

- [ ] インライン JavaScript があれば外部ファイルに移動
- [ ] インライン CSS があれば外部ファイルに移動
- [ ] セマンティック HTML 要素の利用状況を確認（過度な `<div>` ネストの解消）
- [ ] ブラウザで視覚的確認
- [ ] フロントエンドテスト実行して確認

---

## Phase 10: ドキュメント・設定の整理

### 10-1. `pyproject.toml` の充実
> **事前確認**: `pyproject.toml` の現在の内容を確認

- [ ] `[project]` セクションに name, version, description, dependencies を追加
- [ ] `[tool.setuptools]` でパッケージ検出設定を追加
- [ ] `[tool.ruff]` や `[tool.mypy]` 等の静的解析ツール設定を追加（任意）

### 10-2. Dockerfile の整理
> **事前確認**: `Dockerfile` を全体読み取り

- [ ] ビルドキャッシュの効率を確認
- [ ] 不要なレイヤー統合の余地を確認
- [ ] 実施する場合は Docker ビルドテスト

---

## 実行ルール

### 各ステップの実行手順

1. **事前確認**: 該当ファイルを `Read` で読み取り、現在のコード状態を把握する
2. **変更計画**: 変更対象の全ファイルと具体的な変更内容を列挙する
3. **参照調査**: `Grep` で変更対象の関数/変数/import の全参照箇所を特定する
4. **実行**: 変更を適用する
5. **テスト**: 全テストを実行しパスを確認する
6. **コミット**: フェーズごと or ステップごとにコミット

### ハルシネーション防止チェックリスト

- [ ] ファイルの存在確認なしに新ファイルを参照しない
- [ ] 関数名/変数名は `Grep` で実在を確認してから変更する
- [ ] import パスは現在のディレクトリ構造と照合する
- [ ] 分割方針は「案」として記載し、ファイル読み取り後に最終決定する
- [ ] 新しいパターンを導入する前に既存パターンとの整合性を確認する
- [ ] テスト実行で赤くなった場合は原因を特定してから次に進む

### 優先度

| 優先度 | フェーズ | 状態 |
|--------|---------|------|
| **高** | Phase 1 (パッケージ構造) | ✅ 完了 |
| **高** | Phase 2 (命名修正) | ✅ 完了 |
| **高** | Phase 3 (ファイル分割) | ✅ 完了 |
| **中** | Phase 4 (状態管理) | ✅ 完了 |
| **中** | Phase 5 (定数命名修正) | ✅ 完了 |
| **低** | Phase 6 (遅延 import 統一) | ✅ 完了 |
| **低** | Phase 7 (CSS 分割) | ✅ 完了 |
| **低** | Phase 8 (テスト構造改善) | ✅ 完了 |
| **低** | Phase 9-10 | 未着手 |
