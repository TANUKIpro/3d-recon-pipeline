# clip2mesh リファクタリング計画

> **目的**: コードの保守性・可読性・拡張性を向上させる。動作を変えずに構造を改善する。
>
> **原則**:
> - 各ステップは独立してコミット可能な単位とする
> - 既存テスト (pytest + vitest) が全てパスすることを各ステップ完了の条件とする
> - ハルシネーション防止: 各ステップの「事前確認」で現在のコード状態を必ず読み取ってから作業する

---

## Phase 0: 事前準備（リファクタリング前の基盤整備）

### 0-1. テストの実行確認
- [ ] `npm test` でフロントエンドテスト全パスを確認
- [ ] `docker compose -f docker-compose.test.yml run --rm test` で Python テスト全パスを確認（GPU 不要テストのみ）
- [ ] 失敗テストがあれば先に修正する（リファクタリングとは別コミット）

### 0-2. リファクタリング用ブランチ作成
- [ ] `git checkout -b refactor/phase-X` でフェーズごとにブランチを作成
- [ ] 各フェーズ完了時に main へ PR & マージ

---

## Phase 1: Python パッケージ構造の整備

**目標**: `sys.path.insert` ハックを排除し、正規の Python パッケージインポートに統一する

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

## Phase 2: `object_store.py` の公開 API 命名修正

**目標**: `_` prefix の関数が外部から import されている矛盾を解消する

### 2-1. 公開関数の `_` prefix 除去
> **事前確認**: `scripts/dashboard/object_store.py` を読み、全関数定義を確認。`app.py` からの import 一覧と照合

**対象関数（`app.py` から import されている `_` 付き関数）:**
- `_infer_resume_stage` → `infer_resume_stage`
- `_list_objects` → `list_objects`
- `_list_preview_files` → `list_preview_files`
- `_object_dir` → `object_dir`
- `_prepare_object_output_dir` → `prepare_object_output_dir`
- `_reset_outputs_from_stage` → `reset_outputs_from_stage`
- `_resolve_output_root` → `resolve_output_root`
- `_safe_json_load` → `safe_json_load`
- `_sanitize_object_name` → `sanitize_object_name`
- `_suggest_object_name` → `suggest_object_name`
- `_summarize_object` → `summarize_object`
- `_validate_object_name` → `validate_object_name`
- `_validate_resume_prerequisites` → `validate_resume_prerequisites`
- `_write_object_meta` → `write_object_meta`

- [x] `object_store.py` 内の関数名からリーディング `_` を除去
- [x] `app.py`, `test_object_store.py`, `test_app_endpoints.py`, `TEST_PLAN.md` の全参照箇所を一括置換
- [x] 内部専用ヘルパー（`_utc_iso`, `_stage_completion_flags`, `_latest_update_ts`, `_objects_root`）は `_` のまま維持
- [x] Dockerで全テスト実行して確認（469 passed）

---

## Phase 3: 巨大ファイルの分割

**目標**: 800 行超のファイルを論理的な単位に分割し、各ファイルの責務を明確にする

### 3-1. `stage_texture_bake.py`（2910 行, 58 関数）の分割
> **事前確認**: ファイル全体を読み、関数を以下のカテゴリに分類する

**分割方針（案 — 事前確認後に調整）:**

| 新ファイル | 責務 | 含める関数群 |
|---|---|---|
| `stage_texture_bake.py` | 公開エントリ + orchestration | `bake_texture()`, トップレベルオーケストレーション |
| `scripts/texture/intrinsics.py` | カメラ内部パラメータ推定 | `_estimate_intrinsics()` 関連 |
| `scripts/texture/uv_atlas.py` | UV アトラス生成 | `_run_texture_atlas()` 関連 |
| `scripts/texture/view_assign.py` | ビュー割り当て | `_assign_view_per_face()`, conflict 関連 |
| `scripts/texture/seam_blend.py` | シームブレンディング | seam 系関数群 |
| `scripts/texture/render.py` | テクスチャレンダリング | `_bake_texture()`, rasterize 関連 |

- [ ] ファイルを読んで実際の関数依存関係を確認
- [ ] 上記分割方針を依存関係に基づいて最終決定
- [ ] サブモジュール `scripts/texture/` ディレクトリと `__init__.py` を作成
- [ ] 各関数を適切なファイルに移動
- [ ] `stage_texture_bake.py` 本体は薄いオーケストレーション層として維持（サブモジュールから re-import）
- [ ] 既存の `from stage_texture_bake import ...` が壊れないよう `stage_texture_bake.py` に re-export を設置
- [ ] Dockerで全テスト実行して確認

### 3-2. `stage_contact_hole_repair.py`（2188 行, 70 関数）の分割
> **事前確認**: ファイル全体を読み、関数を以下のカテゴリに分類する

**分割方針（案）:**

| 新ファイル | 責務 |
|---|---|
| `stage_contact_hole_repair.py` | 公開エントリ |
| `scripts/repair/boundary.py` | 境界ループ検出 |
| `scripts/repair/triangulate.py` | ホール三角形化 |
| `scripts/repair/candidate.py` | 候補評価 |

- [ ] ファイルを読んで実際の関数依存関係を確認
- [ ] 分割方針を最終決定
- [ ] サブモジュール `scripts/repair/` ディレクトリと `__init__.py` を作成
- [ ] 各関数を移動し re-export を設置
- [ ] Dockerで全テスト実行して確認

### 3-3. `app.py`（975 行）のルーター分割
> **事前確認**: `app.py` を読み、エンドポイントをカテゴリ別に分類する

**分割方針:**

| 新ファイル | 責務 |
|---|---|
| `scripts/dashboard/app.py` | FastAPI app 初期化 + ルーターの include |
| `scripts/dashboard/routes/pipeline.py` | パイプライン制御 API (`/api/pipeline/*`) |
| `scripts/dashboard/routes/sam2.py` | SAM2 API (`/api/sam2/*`) |
| `scripts/dashboard/routes/preview.py` | プレビュー API (`/api/preview/*`) |
| `scripts/dashboard/routes/mesh.py` | メッシュ修復 + ポストプロセス API (`/api/mesh-repair/*`, `/api/mesh/*`) |
| `scripts/dashboard/routes/verification.py` | 検証 API (`/api/verification/*`) |

- [ ] `app.py` を読んでエンドポイント一覧を確認
- [ ] `APIRouter` を使って各ルートファイルに分離
- [ ] `app.py` は `app = FastAPI()` + `app.include_router(...)` のみに
- [ ] 共有状態（`session`, `sam2_service`, `log_broadcaster`）の受け渡し方法を決定（依存性注入 or モジュールグローバル）
- [ ] Dockerで全テスト実行して確認

### 3-4. `pipeline_runner.py`（1083 行）の分割
> **事前確認**: ファイルを読み、責務を確認

**分割方針:**

| 新ファイル | 責務 |
|---|---|
| `pipeline_runner.py` | `run_pipeline()` メインフロー |
| `scripts/dashboard/stage_dispatch.py` | ステージ実行ディスパッチロジック |

- [ ] ファイルを読んで分割が妥当かを判断（1083 行なら分割不要の可能性もあり）
- [ ] 分割する場合は実行し、全テスト確認

### 3-5. フロントエンド巨大ファイルの分割

#### `preview.js`（1741 行）
> **事前確認**: ファイルを読み、責務を確認

**分割方針（案）:**

| 新ファイル | 責務 |
|---|---|
| `preview.js` | メイン export + 初期化 |
| `js/preview/scene-manager.js` | three.js シーン管理 |
| `js/preview/loaders.js` | PLY/OBJ ローダー |
| `js/preview/mesh-repair-viz.js` | メッシュ修復ループの可視化 |
| `js/preview/gallery.js` | フレームギャラリー |

- [ ] ファイルを読んで実際の構造を確認
- [ ] 分割方針を最終決定し実行
- [ ] `index.html` の `<script>` タグを更新
- [ ] フロントエンドテスト実行して確認

#### `config-panel.js`（1566 行）
> **事前確認**: ファイルを読み、責務を確認

**分割方針（案）:**

| 新ファイル | 責務 |
|---|---|
| `config-panel.js` | メイン export + 初期化 |
| `js/config/video-selector.js` | ビデオ/オブジェクト選択 |
| `js/config/stage-params.js` | ステージごとのパラメータ UI |
| `js/config/validators.js` | 入力バリデーション |

- [ ] ファイルを読んで実際の構造を確認
- [ ] 分割方針を最終決定し実行
- [ ] フロントエンドテスト実行して確認

---

## Phase 4: `app.py` の共有状態管理の改善

**目標**: グローバル変数によるモジュール間結合を緩和する

### 4-1. 依存コンテナの導入
> **事前確認**: `app.py` のグローバル変数 (`session`, `sam2_service`, `log_broadcaster`, `INPUT_DIR`, `OUTPUT_DIR`) の利用箇所を確認

- [ ] `scripts/dashboard/dependencies.py` を作成し、共有状態をまとめる
  ```python
  # 案（事前確認後に決定）
  class AppState:
      session: PipelineSession
      sam2_service: SAM2Service
      log_broadcaster: LogBroadcaster | None
      input_dir: str
      output_dir: str
  ```
- [ ] 各ルーターで FastAPI `Depends()` を使ってアクセス
- [ ] `app.py` のグローバル変数を `AppState` インスタンスに置換
- [ ] Dockerで全テスト実行して確認

---

## Phase 5: `config_defaults.py` の定数整理

**目標**: public 定数と internal 定数の命名を一貫させ、構造化する

### 5-1. 定数のグルーピング
> **事前確認**: `scripts/config_defaults.py` を全体読み取り、定数を分類

- [ ] ファイルを読んでセクションごとの定数一覧を作成
- [ ] `_` prefix 付き定数（internal）と prefix なし定数（public/user-configurable）の区別が正しいか確認
- [ ] 一貫性のない命名があれば修正案を作成
- [ ] 外部からの利用箇所を `grep` で全列挙してから名前変更を実行
- [ ] Dockerで全テスト実行して確認

---

## Phase 6: `stage_wrappers.py` の遅延 import パターン統一

**目標**: 各ラッパー関数内に散らばる遅延 import を統一パターンにする

### 6-1. パターンの確認と統一
> **事前確認**: `scripts/dashboard/stage_wrappers.py` を読み、各関数内の import パターンを確認

- [ ] 現状の遅延 import パターンを全列挙
- [ ] 統一パターンを決定（例: 各関数冒頭で import、エラーハンドリング統一）
- [ ] ステージラッパー間で重複している import を整理
- [ ] Dockerで全テスト実行して確認

---

## Phase 7: フロントエンド CSS の分割

**目標**: `style.css`（1702 行）を論理単位に分割

### 7-1. CSS の分割
> **事前確認**: `style.css` を読み、セクションコメントとセレクタを確認

**分割方針（案）:**

| 新ファイル | 責務 |
|---|---|
| `static/css/base.css` | リセット、CSS変数、タイポグラフィ |
| `static/css/layout.css` | メインレイアウト、グリッド |
| `static/css/pipeline.css` | パイプライン UI（ステージピル等） |
| `static/css/sam2.css` | SAM2 キャンバス・検証 UI |
| `static/css/preview.css` | 3D プレビュー |
| `static/css/config.css` | 設定パネル |
| `static/css/log.css` | ログビューア |

- [ ] `style.css` のセクション構造を確認
- [ ] 分割方針を最終決定
- [ ] `index.html` の `<link>` タグを更新
- [ ] ブラウザで視覚的確認
- [ ] フロントエンドテスト実行して確認

---

## Phase 8: テスト構造の改善

**目標**: テストヘルパーの重複を排除し、カバレッジを可視化する

### 8-1. Python テストの conftest 整理
> **事前確認**: `tests/conftest.py` と `tests/dashboard/backend/` 内の各テストファイルの fixture を確認

- [ ] 重複する fixture やモックパターンを確認
- [ ] 共通 fixture を `conftest.py` に集約
- [ ] Dockerで全テスト実行して確認

### 8-2. フロントエンドテストヘルパーの整理
> **事前確認**: `tests/dashboard/frontend/helpers/` 内のファイルを確認

- [ ] `dom-factory.js`, `fetch-mock.js`, `ws-mock.js`, `three-stub.js` の利用パターンを確認
- [ ] 共通パターンの抽出が可能か検討
- [ ] 実施する場合はリファクタリングしてテスト実行

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

| 優先度 | フェーズ | 理由 |
|--------|---------|------|
| **高** | Phase 1 (パッケージ構造) | 全ての後続変更の基盤 |
| **高** | Phase 2 (命名修正) | Phase 1 と同時に実施可能 |
| **高** | Phase 3 (ファイル分割) | 最も可読性に影響 |
| **中** | Phase 4 (状態管理) | Phase 3 のルーター分割と連動 |
| **中** | Phase 5 (定数整理) | 単独で実施可能 |
| **低** | Phase 6-10 | 品質向上だが緊急度は低い |
