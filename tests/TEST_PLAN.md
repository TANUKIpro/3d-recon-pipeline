# clip2mesh Dashboard テスト計画

現在の動作を完全に保証するためのテスト項目一覧。
既存テストでカバー済みの項目には `[済]` を付与。未実装は `[ ]`。

---

## 1. Backend: State (`state.py`)

### 1.1 Enum / 定数

| # | テスト項目 | 状態 |
|---|-----------|------|
| 1.1.1 | `StageStatus` の値が JSON コントラクト通り (`pending`, `running`, `complete`, `failed`, `interactive`) | [済] |
| 1.1.2 | `StageStatus` の `str()` がそのまま値を返す | [済] |
| 1.1.3 | `PipelineStage` の IntEnum 順序 (IDLE=0 → COMPLETE=9) | [済] |
| 1.1.4 | `STAGE_LABELS` が Stage 1–8 すべてをカバー | [済] |
| 1.1.5 | `STAGE_OUTPUT_FILES` が Stage 2–8 すべてをカバー | [済] |

### 1.2 PipelineConfig

| # | テスト項目 | 状態 |
|---|-----------|------|
| 1.2.1 | デフォルト構築で全フィールドに値がある | [済] |
| 1.2.2 | `from_dict()` → `to_dict()` のラウンドトリップ | [済] |
| 1.2.3 | `to_dict()` が全フィールドを含む | [済] |
| 1.2.4 | `from_dict()` が未知キーを無視する | [済] |
| 1.2.5 | `from_dict()` → `to_dict()` で全値が保持される | [済] |

### 1.3 StageInfo

| # | テスト項目 | 状態 |
|---|-----------|------|
| 1.3.1 | デフォルト値 (status=PENDING, progress=0, error=None 等) | [済] |

### 1.4 PipelineSession — 初期化・リセット

| # | テスト項目 | 状態 |
|---|-----------|------|
| 1.4.1 | 初期化で stages[1]–stages[8] がすべて存在 | [済] |
| 1.4.2 | 初期化後の前提条件 (running=False, cancelled=False, ws_clients=[] 等) | [済] |
| 1.4.3 | `reset()` でランタイムフィールドがクリアされる | [済] |
| 1.4.4 | `reset()` 後に全ステージが PENDING | [済] |
| 1.4.5 | `reset()` でアーティファクトパスがクリアされる | [済] |
| 1.4.6 | `reset()` でキャンセル状態がクリアされる | [済] |
| 1.4.7 | `reset()` でメッシュ修復状態がクリアされる | [済] |
| 1.4.8 | `reset()` で SAM2 状態がクリアされる | [済] |
| 1.4.9 | `reset()` で asyncio.Event が新規インスタンスに置換される | [ ] |
| 1.4.10 | `reset()` で `_active_processes` がクリアされる | [ ] |

### 1.5 PipelineSession — ステージライフサイクル

| # | テスト項目 | 状態 |
|---|-----------|------|
| 1.5.1 | `stage_start()` で status=RUNNING, start_time 設定 | [済] |
| 1.5.2 | `stage_complete()` で status=COMPLETE, progress=100 | [済] |
| 1.5.3 | `stage_failed()` で status=FAILED, error 記録 | [済] |
| 1.5.4 | `stage_interactive()` で status=INTERACTIVE | [済] |
| 1.5.5 | `stage_progress()` で progress が 0–100 にクランプ | [済] |
| 1.5.6 | `stage_progress()` で checkpoint_id 更新 | [済] |
| 1.5.7 | `stage_progress()` で progress=None 時に既存値を上書きしない | [済] |

### 1.6 PipelineSession — ステージ遷移確認ゲート

| # | テスト項目 | 状態 |
|---|-----------|------|
| 1.6.1 | `require_next_stage_confirmation()` で confirmation フィールドが設定される | [済] |
| 1.6.2 | `clear_next_stage_confirmation()` で confirmation フィールドがリセットされる | [済] |
| 1.6.3 | `clear_next_stage_confirmation()` で event が新規インスタンスに置換される | [ ] |

### 1.7 PipelineSession — キャンセル

| # | テスト項目 | 状態 |
|---|-----------|------|
| 1.7.1 | `request_cancel()` で cancel_requested=True, cancel_event.is_set() | [済] |
| 1.7.2 | `request_cancel(force=True)` で cancel_force=True | [済] |
| 1.7.3 | `clear_cancel()` で全キャンセルフラグがリセットされる | [済] |

### 1.8 PipelineSession — プロセス管理

| # | テスト項目 | 状態 |
|---|-----------|------|
| 1.8.1 | `register_active_process()` でプロセスが集合に追加される | [済] |
| 1.8.2 | `unregister_active_process()` でプロセスが集合から削除される | [済] |
| 1.8.3 | `terminate_active_processes()` で SIGTERM → SIGKILL の順で呼ばれる | [済] |
| 1.8.4 | 終了済みプロセスの terminate はスキップされる | [済] |

### 1.9 PipelineSession — メッシュ修復

| # | テスト項目 | 状態 |
|---|-----------|------|
| 1.9.1 | `set_mesh_repair_candidates()` で候補と分析データが設定される | [済] |
| 1.9.2 | `clear_mesh_repair_candidates()` で状態がリセットされる | [済] |
| 1.9.3 | `hydrate_from_output_dir()` で interactive メッシュ修復状態がクリアされる | [済] |

### 1.10 PipelineSession — overall_progress / to_status_dict

| # | テスト項目 | 状態 |
|---|-----------|------|
| 1.10.1 | 全ステージ 0% → 全体 0% | [済] |
| 1.10.2 | 全ステージ 100% → 全体 100% | [済] |
| 1.10.3 | 一部ステージのみ進行 → 加重平均 | [済] |
| 1.10.4 | `to_status_dict()` のトップレベルキーが全て揃う | [済] |
| 1.10.5 | `to_status_dict()` の stages キーが文字列 ("1"–"8") | [済] |
| 1.10.6 | 実行中の elapsed が経過時間を返す | [済] |
| 1.10.7 | アイドル時の elapsed が None を返す | [済] |
| 1.10.8 | `to_status_dict()` に `next_stage_confirmation` が含まれる | [ ] |
| 1.10.9 | `to_status_dict()` に `mesh_repair_ready` が含まれる | [ ] |

### 1.11 detect_stage_outputs / hydrate_from_output_dir

| # | テスト項目 | 状態 |
|---|-----------|------|
| 1.11.1 | 空ディレクトリでは全ステージ未完了 | [済] |
| 1.11.2 | frames/ のみ → Stage 1 完了 | [済] |
| 1.11.3 | Stage 3 判定 (object.ply + masks/) | [済] |
| 1.11.4 | テストフィクスチャで全ステージ完了 | [済] |
| 1.11.5 | `hydrate_from_output_dir()` で frames_dir パス設定 | [済] |
| 1.11.6 | frames なしで frames_dir=None | [済] |
| 1.11.7 | メッシュ優先度: repaired > wrapped > base | [済] |
| 1.11.8 | current_stage が最後の完了ステージから推定される | [済] |
| 1.11.9 | 戻り値に期待されるキーが含まれる | [済] |
| 1.11.10 | Stage 6 完了判定 (wrapped mesh 存在時) | [済] |
| 1.11.11 | Stage 7 完了判定 (repaired mesh 存在時) | [済] |

---

## 2. Backend: Configuration (`configuration.py`)

### 2.1 パーサユーティリティ

| # | テスト項目 | 状態 |
|---|-----------|------|
| 2.1.1 | `parse_int`: 有効な整数文字列 | [済] |
| 2.1.2 | `parse_int`: float 文字列は失敗 | [済] |
| 2.1.3 | `parse_int`: None → fallback | [済] |
| 2.1.4 | `parse_int`: 空文字列 → fallback | [済] |
| 2.1.5 | `parse_int`: 不正文字列 → fallback | [済] |
| 2.1.6 | `parse_int`: 負の値 | [済] |
| 2.1.7 | `parse_int`: bool は int に変換 | [済] |
| 2.1.8 | `parse_float`: 有効な float 文字列 | [済] |
| 2.1.9 | `parse_float`: None → fallback | [済] |
| 2.1.10 | `parse_float`: 科学的記法 | [済] |
| 2.1.11 | `parse_choice`: 有効な選択肢 | [済] |
| 2.1.12 | `parse_choice`: 無効な選択肢 → fallback | [済] |
| 2.1.13 | `parse_choice`: None → fallback | [済] |
| 2.1.14 | `parse_choice`: 空白トリム | [済] |
| 2.1.15 | `parse_bool`: true 系 ("1", "true", "yes", "on", "y") | [済] |
| 2.1.16 | `parse_bool`: false 系 ("0", "false", "no", "off", "n") | [済] |
| 2.1.17 | `parse_bool`: None → fallback | [済] |
| 2.1.18 | `parse_bool`: bool 直接受け取り | [済] |
| 2.1.19 | `env_int`: 環境変数辞書から読み取り | [済] |
| 2.1.20 | `env_int`: キー不在 → fallback | [済] |
| 2.1.21 | `env_float`: 読み取り / fallback | [済] |
| 2.1.22 | `env_bool`: true/false/不在 | [済] |

### 2.2 build_pipeline_config

| # | テスト項目 | 状態 |
|---|-----------|------|
| 2.2.1 | DiffCD 値が raw からそのまま適用される | [済] |
| 2.2.2 | 数値フィールドのクランプと正規化 | [済] |
| 2.2.3 | texture_size=0 (auto モード) が保持される | [済] |
| 2.2.4 | texture_view_assign_mode のデフォルト値 | [済] |
| 2.2.5 | texture_view_assign_mode が "region_gc" を受け付ける | [済] |
| 2.2.6 | texture_quality_boost のデフォルトが false | [済] |
| 2.2.7 | texture_quality_boost が true を受け付ける | [済] |
| 2.2.8 | mesh_wrap / repair デフォルトが contact-friendly プロファイルに一致 | [済] |
| 2.2.9 | mesh_repair フィールドのパースとクランプ | [済] |
| 2.2.10 | denoise_preset="custom" で raw 値が保持される | [済] |
| 2.2.11 | denoise_preset 有効値でプリセットが適用される | [済] |
| 2.2.12 | denoise_preset 不明値で fallback | [済] |
| 2.2.13 | classical_preset="custom" で保持 | [済] |
| 2.2.14 | classical_preset 有効値で適用 | [済] |
| 2.2.15 | pi3x_frame_target が max_frames 以下にクランプ | [済] |
| 2.2.16 | meshwrap_poisson_depth の最小値 6 | [済] |
| 2.2.17 | meshwrap_poisson_scale の最小値 1 | [済] |
| 2.2.18 | mesh_repair 各フィールドのクランプ | [済] |
| 2.2.19 | 環境変数による mesh_method オーバーライド | [済] |
| 2.2.20 | auto_accept が parse_bool 経由でパースされる | [済] |
| 2.2.21 | mesh_method が parse_choice 経由でパースされる | [済] |
| 2.2.22 | denoise プリセットが全キーを持つ | [済] |
| 2.2.23 | ground_plane_enabled のパースと環境変数反映 | [ ] |
| 2.2.24 | frame_interval / max_frames / pixel_limit の環境変数反映 | [ ] |
| 2.2.25 | 全フィールドがデフォルト値のみで config 構築可能 (raw={}) | [ ] |

---

## 3. Backend: Checkpoints (`checkpoints.py`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 3.1 | `mesh_method_key`: "diffcd" (小文字) | [済] |
| 3.2 | `mesh_method_key`: 大文字小文字不問 | [済] |
| 3.3 | `mesh_method_key`: None → "poisson" | [済] |
| 3.4 | `mesh_method_key`: 不明値 → "poisson" | [済] |
| 3.5 | `checkpoint_specs`: Stage 1 がタプルを返す | [済] |
| 3.6 | `checkpoint_specs`: Stage 1 の spec が必須キーを持つ | [済] |
| 3.7 | `checkpoint_specs`: Stage 5 poisson 分岐 | [済] |
| 3.8 | `checkpoint_specs`: Stage 5 diffcd 分岐 | [済] |
| 3.9 | `checkpoint_specs`: 無効ステージ → 空タプル | [済] |
| 3.10 | `first_checkpoint_id`: 各ステージの最初の ID | [済] |
| 3.11 | `resolve_checkpoint_id`: "starting" → first | [済] |
| 3.12 | `resolve_checkpoint_id`: "complete" → last | [済] |
| 3.13 | `resolve_checkpoint_id`: パターンマッチ | [済] |
| 3.14 | `resolve_checkpoint_id`: "waiting for next-stage" → last | [済] |
| 3.15 | `resolve_checkpoint_id`: 空/None → current 維持 | [済] |
| 3.16 | `checkpoint_cleanup_plan`: 既知チェックポイント | [済] |
| 3.17 | `checkpoint_cleanup_plan`: 不明チェックポイント → fallback | [済] |
| 3.18 | `cleanup_checkpoint_outputs`: ディレクトリ削除 | [済] |
| 3.19 | `cleanup_checkpoint_outputs`: ファイル削除 | [済] |
| 3.20 | `cleanup_checkpoint_outputs`: 存在しないパスは安全 | [済] |
| 3.21 | `cleanup_checkpoint_outputs`: パストラバーサル (`..`) ブロック | [済] |
| 3.22 | `cleanup_checkpoint_outputs`: 絶対パスブロック | [済] |
| 3.23 | `cleanup_checkpoint_outputs`: 戻り値構造 | [済] |

---

## 4. Backend: Log Capture (`log_capture.py`)

### 4.1 stage_log_scope

| # | テスト項目 | 状態 |
|---|-----------|------|
| 4.1.1 | ContextVar の設定と復元 | [済] |
| 4.1.2 | ネスト時の正しい値管理 | [済] |
| 4.1.3 | stage=None の挙動 | [済] |

### 4.2 StreamCapture

| # | テスト項目 | 状態 |
|---|-----------|------|
| 4.2.1 | write がオリジナルとコールバック両方に転送 | [済] |
| 4.2.2 | 空文字列は no-op | [済] |
| 4.2.3 | write が文字列長を返す | [済] |
| 4.2.4 | flush がオリジナルに委譲 | [済] |
| 4.2.5 | fileno がオリジナルに委譲 | [済] |
| 4.2.6 | isatty が false を返す | [済] |

### 4.3 LogBroadcaster — install/uninstall

| # | テスト項目 | 状態 |
|---|-----------|------|
| 4.3.1 | install で stdout/stderr が置換される | [済] |
| 4.3.2 | uninstall でオリジナルが復元される | [済] |
| 4.3.3 | 二重 install が安全 | [済] |
| 4.3.4 | 二重 uninstall が安全 | [済] |

### 4.4 LogBroadcaster — _on_write

| # | テスト項目 | 状態 |
|---|-----------|------|
| 4.4.1 | メッセージ構造 (type, stream, text) | [済] |
| 4.4.2 | ContextVar からの stage 取得 | [済] |
| 4.4.3 | resolver からの stage fallback | [済] |
| 4.4.4 | RuntimeError 抑制 (ループ閉鎖時) | [済] |
| 4.4.5 | resolver 例外時に stage なし | [済] |

### 4.5 LogBroadcaster — drain

| # | テスト項目 | 状態 |
|---|-----------|------|
| 4.5.1 | 全クライアントがメッセージを受信 | [済] |
| 4.5.2 | 壊れたクライアントが除去される | [済] |

---

## 5. Backend: SAM2 Service (`sam2_service.py`)

### 5.1 初期化

| # | テスト項目 | 状態 |
|---|-----------|------|
| 5.1.1 | 初期状態で initialized=False | [済] |
| 5.1.2 | initialize() でメタデータ返却 & initialized=True | [済] |
| 5.1.3 | 二重 initialize でセッション置換 | [済] |

### 5.2 add_click

| # | テスト項目 | 状態 |
|---|-----------|------|
| 5.2.1 | 未初期化で例外 | [済] |
| 5.2.2 | クリックでポイント・ラベル追加 | [済] |
| 5.2.3 | PNG バイトが返る | [済] |
| 5.2.4 | ground モードで ground リストに格納 | [済] |
| 5.2.5 | 座標の正規化 (0–1 → ピクセル座標変換) | [ ] |

### 5.3 undo_click

| # | テスト項目 | 状態 |
|---|-----------|------|
| 5.3.1 | 未初期化で例外 | [済] |
| 5.3.2 | 最後のクリックが削除される | [済] |
| 5.3.3 | 空リストで安全 | [済] |
| 5.3.4 | ground モードでの undo が ground リストのみ影響 | [ ] |

### 5.4 clear_clicks

| # | テスト項目 | 状態 |
|---|-----------|------|
| 5.4.1 | 未初期化で例外 | [済] |
| 5.4.2 | 全クリックが削除される | [済] |
| 5.4.3 | ground clear で object クリックが保持される | [済] |
| 5.4.4 | predictor.reset_state が呼ばれる | [ ] |

### 5.5 propagate_and_save

| # | テスト項目 | 状態 |
|---|-----------|------|
| 5.5.1 | 未初期化で例外 | [済] |
| 5.5.2 | クリックなしで例外 | [済] |
| 5.5.3 | 正常実行で (mask_dir, ground_mask_dir) タプル返却 | [済] |
| 5.5.4 | progress_callback が各フレームで呼ばれる | [ ] |
| 5.5.5 | マスクファイルがディスクに保存される | [ ] |
| 5.5.6 | 既存マスクファイルがクリアされてから保存 | [ ] |

### 5.6 release

| # | テスト項目 | 状態 |
|---|-----------|------|
| 5.6.1 | release_model が呼ばれる | [済] |
| 5.6.2 | session が None になる | [済] |
| 5.6.3 | 未初期化での release は no-op | [済] |
| 5.6.4 | current_mask がクリアされる | [済] |

### 5.7 get_frame / get_mask

| # | テスト項目 | 状態 |
|---|-----------|------|
| 5.7.1 | 未初期化で例外 | [済] |
| 5.7.2 | 範囲外インデックスで例外 | [済] |
| 5.7.3 | JPEG バイトが返る | [済] |
| 5.7.4 | `get_mask_png` が存在するマスクの PNG バイトを返す | [ ] |
| 5.7.5 | `get_mask_png` が存在しないマスクで None を返す | [ ] |

### 5.8 スレッドセーフティ

| # | テスト項目 | 状態 |
|---|-----------|------|
| 5.8.1 | 並行 add_click でデッドロックしない | [済] |

### 5.9 セグメンテーションモード

| # | テスト項目 | 状態 |
|---|-----------|------|
| 5.9.1 | デフォルトモードが "object" | [ ] |
| 5.9.2 | モード切替で segmentation_mode プロパティが更新される | [ ] |
| 5.9.3 | has_ground_clicks が ground クリック有無を正しく返す | [ ] |

---

## 6. Backend: Pipeline Runner (`pipeline_runner.py`)

### 6.1 broadcast / broadcast_to_clients

| # | テスト項目 | 状態 |
|---|-----------|------|
| 6.1.1 | 全クライアントがメッセージ受信 | [済] |
| 6.1.2 | 壊れた接続が除去される | [済] |
| 6.1.3 | 空クライアントリストで安全 | [済] |

### 6.2 _build_stage_progress_payload

| # | テスト項目 | 状態 |
|---|-----------|------|
| 6.2.1 | 必須キー (type, stage, progress, detail, checkpoint_id) | [済] |
| 6.2.2 | session の progress/checkpoint が更新される | [済] |
| 6.2.3 | checkpoint_id が resolve される | [済] |

### 6.3 _CancelledError

| # | テスト項目 | 状態 |
|---|-----------|------|
| 6.3.1 | デフォルトメッセージ | [済] |
| 6.3.2 | Exception を継承 | [済] |

### 6.4 ヘルパー関数

| # | テスト項目 | 状態 |
|---|-----------|------|
| 6.4.1 | `_safe_current_stage`: 有効ステージ返却 | [済] |
| 6.4.2 | `_safe_current_stage`: IDLE/COMPLETE → None | [済] |
| 6.4.3 | `_require_file`: None / 空 / 存在しないファイルで例外 | [済] |
| 6.4.4 | `_require_file`: 存在するファイルで OK | [済] |
| 6.4.5 | `_require_dir`: None / 存在しない / 空ディレクトリで例外 | [済] |
| 6.4.6 | `_require_dir`: ファイルありで OK | [済] |

### 6.5 _check_cancelled

| # | テスト項目 | 状態 |
|---|-----------|------|
| 6.5.1 | cancelled フラグで例外 | [済] |
| 6.5.2 | cancel_requested フラグで例外 | [済] |
| 6.5.3 | cancel_event.is_set() で例外 | [済] |
| 6.5.4 | 通常状態で例外なし | [済] |

### 6.6 _wait_for_next_stage_confirmation

| # | テスト項目 | 状態 |
|---|-----------|------|
| 6.6.1 | auto_accept=True で即時通過 | [済] |
| 6.6.2 | 待機中のキャンセルで _CancelledError | [済] |
| 6.6.3 | broadcast メッセージ構造 (required/cleared) | [ ] |
| 6.6.4 | session の confirmation フィールドが設定・クリアされる | [ ] |

### 6.7 _mesh_method_label

| # | テスト項目 | 状態 |
|---|-----------|------|
| 6.7.1 | "diffcd" → 適切なラベル | [済] |
| 6.7.2 | "poisson" → 適切なラベル | [済] |

### 6.8 run_pipeline — ステージフロー (統合テスト)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 6.8.1 | Stage 1–8 の順次実行 (全モック) | [ ] |
| 6.8.2 | resume_from_stage > 1 での途中開始 | [ ] |
| 6.8.3 | キャンセル時の pipeline_error broadcast (reason_code="cancelled") | [ ] |
| 6.8.4 | キャンセル後の hydrate_from_output_dir 呼び出し | [ ] |
| 6.8.5 | 例外時の pipeline_error broadcast | [ ] |
| 6.8.6 | finally での SAM2 release 呼び出し | [ ] |
| 6.8.7 | SAM2 redo ループ (approve=false → 再初期化 → approve=true) | [ ] |
| 6.8.8 | メッシュ修復 interactive フロー (候補表示 → ユーザ選択 → 修復) | [ ] |
| 6.8.9 | auto_accept モードでの SAM2/confirmation 自動通過 | [ ] |
| 6.8.10 | ground_plane_enabled=true 時の ground segmentation フロー | [ ] |
| 6.8.11 | ground skip イベントでの ground フェーズスキップ | [ ] |

### 6.9 _run_stage

| # | テスト項目 | 状態 |
|---|-----------|------|
| 6.9.1 | stage_start → fn 実行 → stage_complete の順序 | [ ] |
| 6.9.2 | fn 内例外で stage_failed + 再 raise | [ ] |
| 6.9.3 | progress_cb / cancel_cb が fn に渡される | [ ] |
| 6.9.4 | stage_log_scope 内での実行 | [ ] |

### 6.10 _make_progress_cb / _make_cancel_cb

| # | テスト項目 | 状態 |
|---|-----------|------|
| 6.10.1 | progress_cb が loop.call_soon_threadsafe を使用 | [ ] |
| 6.10.2 | cancel_cb がキャンセル状態で例外 | [ ] |

---

## 7. Backend: Object Store (`object_store.py`)

### 7.1 名前処理

| # | テスト項目 | 状態 |
|---|-----------|------|
| 7.1.1 | `_sanitize_object_name`: 通常名そのまま | [済] |
| 7.1.2 | `_sanitize_object_name`: スペース → ハイフン | [済] |
| 7.1.3 | `_sanitize_object_name`: スラッシュ → ハイフン | [済] |
| 7.1.4 | `_sanitize_object_name`: 特殊文字置換 | [済] |
| 7.1.5 | `_sanitize_object_name`: 連続ハイフン圧縮 | [済] |
| 7.1.6 | `_sanitize_object_name`: 空文字列 → ValueError | [済] |
| 7.1.7 | `_sanitize_object_name`: 80 文字で切り捨て | [済] |
| 7.1.8 | `_sanitize_object_name`: 先頭末尾のドット/ハイフン除去 | [済] |
| 7.1.9 | `_validate_object_name`: 正常名パス | [済] |
| 7.1.10 | `_validate_object_name`: 空/"."/".."/ "/" / "\\" → ValueError | [済] |
| 7.1.11 | `_suggest_object_name`: ビデオパスから stem 抽出 | [済] |
| 7.1.12 | `_suggest_object_name`: 空パス → "object" | [済] |
| 7.1.13 | `_suggest_object_name`: 特殊文字がサニタイズされる | [済] |

### 7.2 出力管理

| # | テスト項目 | 状態 |
|---|-----------|------|
| 7.2.1 | `STAGE_RESET_PATHS`: Stage 5 リセットに preview mesh が含まれる | [済] |
| 7.2.2 | `RESUME_PREREQUISITES`: Stage 6/7 に mesh 出力が必要 | [済] |
| 7.2.3 | `_reset_outputs_from_stage`: 指定ステージ以降のファイル/ディレクトリ削除 | [ ] |
| 7.2.4 | `_infer_resume_stage`: 最初の未完了ステージを返す | [ ] |
| 7.2.5 | `_validate_resume_prerequisites`: 必要ファイルの存在チェック | [ ] |
| 7.2.6 | `_write_object_meta`: JSON 書き込み・created_at 保持 | [ ] |
| 7.2.7 | `_summarize_object`: メタデータ辞書構築 | [ ] |
| 7.2.8 | `_list_objects`: updated_at 降順ソート | [ ] |

### 7.3 safe_json_load

| # | テスト項目 | 状態 |
|---|-----------|------|
| 7.3.1 | 有効な JSON dict | [済] |
| 7.3.2 | 存在しないファイル → 空 dict | [済] |
| 7.3.3 | 壊れた JSON → 空 dict | [済] |
| 7.3.4 | dict 以外の JSON → 空 dict | [済] |

---

## 8. Backend: App Endpoints (`app.py`)

### 8.1 Pipeline API

| # | テスト項目 | 状態 |
|---|-----------|------|
| 8.1.1 | `GET /api/pipeline/status`: ステータス辞書のフォーマット | [済] |
| 8.1.2 | `POST /api/pipeline/cancel`: 非実行中 → 409 | [済] |
| 8.1.3 | `POST /api/pipeline/cancel`: 実行中 → キャンセルフラグ設定 | [済] |
| 8.1.4 | `POST /api/pipeline/confirm-next`: 非実行中 → 409 | [済] |
| 8.1.5 | `POST /api/pipeline/confirm-next`: 確認待ちなし → 409 | [済] |
| 8.1.6 | `POST /api/pipeline/confirm-next`: event 設定 | [済] |
| 8.1.7 | `GET /api/pipeline/videos`: ビデオ一覧のフォーマット | [ ] |
| 8.1.8 | `GET /api/pipeline/videos`: 対象拡張子のフィルタリング (.mp4, .avi, .mov, .mkv, .webm) | [ ] |
| 8.1.9 | `GET /api/pipeline/objects`: オブジェクト一覧返却 | [ ] |
| 8.1.10 | `GET /api/pipeline/object-info`: 正常返却 | [ ] |
| 8.1.11 | `GET /api/pipeline/object-info`: 不正な名前 → 400 | [ ] |
| 8.1.12 | `GET /api/pipeline/object-info`: 存在しない → 404 | [ ] |
| 8.1.13 | `POST /api/pipeline/load-object`: 正常ロード | [ ] |
| 8.1.14 | `POST /api/pipeline/load-object`: 実行中 → 409 | [ ] |
| 8.1.15 | `POST /api/pipeline/start`: 正常開始 (タスク作成) | [ ] |
| 8.1.16 | `POST /api/pipeline/start`: 実行中 → 409 | [ ] |
| 8.1.17 | `POST /api/pipeline/start`: resume_from_stage 範囲外 → 400 | [ ] |
| 8.1.18 | `POST /api/pipeline/start`: 前提条件未満足 → 400 | [ ] |
| 8.1.19 | `GET /api/pipeline/video-info`: 正常 (fps, total_frames 等) | [ ] |
| 8.1.20 | `GET /api/pipeline/video-info`: INPUT_DIR 外パス → 403 | [済] |
| 8.1.21 | `GET /api/pipeline/pi3x-plan`: 正常返却 | [ ] |

### 8.2 SAM2 API

| # | テスト項目 | 状態 |
|---|-----------|------|
| 8.2.1 | `POST /api/sam2/click`: 未初期化 → 409 | [済] |
| 8.2.2 | `POST /api/sam2/click`: 正常 → PNG レスポンス | [ ] |
| 8.2.3 | `POST /api/sam2/undo`: 正常 → PNG レスポンス | [ ] |
| 8.2.4 | `POST /api/sam2/clear`: 正常 → PNG レスポンス | [ ] |
| 8.2.5 | `POST /api/sam2/confirm`: 未初期化 → 409 | [済] |
| 8.2.6 | `POST /api/sam2/confirm`: event 設定 | [済] |
| 8.2.7 | `POST /api/sam2/approve`: フラグと event 設定 | [済] |
| 8.2.8 | `POST /api/sam2/redo`: フラグクリアと event 設定 | [済] |
| 8.2.9 | `POST /api/sam2/mode`: モード切替 | [ ] |
| 8.2.10 | `GET /api/sam2/mode`: 現在のモード返却 | [ ] |
| 8.2.11 | `POST /api/sam2/skip-ground`: event 設定 | [ ] |
| 8.2.12 | `GET /api/sam2/frame/{idx}`: 正常 → JPEG | [ ] |
| 8.2.13 | `GET /api/sam2/frame/{idx}`: 範囲外 → 404 | [ ] |
| 8.2.14 | `GET /api/sam2/mask/{idx}`: 正常 → PNG | [ ] |
| 8.2.15 | `GET /api/sam2/mask/{idx}`: 存在しない → 404 | [ ] |

### 8.3 Mesh Repair API

| # | テスト項目 | 状態 |
|---|-----------|------|
| 8.3.1 | `GET /api/mesh-repair/candidates`: 非実行中 → 409 | [済] |
| 8.3.2 | `GET /api/mesh-repair/candidates`: 未準備 → 409 | [済] |
| 8.3.3 | `GET /api/mesh-repair/candidates`: 正常 → 候補データ | [済] |
| 8.3.4 | `POST /api/mesh-repair/confirm`: 空選択 (スキップ) 許可 | [済] |
| 8.3.5 | `POST /api/mesh-repair/confirm`: 有効な loop_ids | [済] |
| 8.3.6 | `POST /api/mesh-repair/confirm`: 不明な loop_ids → 400 | [済] |
| 8.3.7 | `POST /api/mesh-repair/confirm`: 重複 ID の除去 | [ ] |

### 8.4 Mesh Post-Process API

| # | テスト項目 | 状態 |
|---|-----------|------|
| 8.4.1 | `POST /api/mesh/postprocess`: 正常実行 → 結果返却 | [ ] |
| 8.4.2 | `POST /api/mesh/postprocess`: method バリデーション ("laplacian"/"taubin") | [ ] |
| 8.4.3 | `POST /api/mesh/postprocess`: iterations クランプ (0–100) | [ ] |
| 8.4.4 | `POST /api/mesh/postprocess`: invalidate_texture=true → ステージ 6–8 リセット | [ ] |
| 8.4.5 | `POST /api/mesh/postprocess`: 実行中かつ非ステージ 5 待ち → 409 | [ ] |

### 8.5 Preview / File API

| # | テスト項目 | 状態 |
|---|-----------|------|
| 8.5.1 | `GET /api/preview/file/{path}`: no-cache ヘッダ | [済] |
| 8.5.2 | `GET /api/preview/file/{path}`: パストラバーサル防止 | [済] |
| 8.5.3 | `GET /api/preview/object-file/{name}/{path}`: 正常ファイル配信 | [ ] |
| 8.5.4 | `GET /api/preview/object-file/{name}/{path}`: パス脱出 → 403 | [済] |
| 8.5.5 | `GET /api/preview/outputs`: プレビュー対象ファイル一覧 | [ ] |
| 8.5.6 | `GET /api/preview/crop-obb`: OBB 計算結果 | [ ] |
| 8.5.7 | `GET /api/verification/frame/{idx}`: マスクオーバーレイ画像 | [ ] |
| 8.5.8 | `GET /api/verification/ground-frame/{idx}`: ground オーバーレイ画像 | [ ] |

### 8.6 Utility API

| # | テスト項目 | 状態 |
|---|-----------|------|
| 8.6.1 | `GET /api/vram`: free_mb が数値または null | [ ] |

### 8.7 WebSocket

| # | テスト項目 | 状態 |
|---|-----------|------|
| 8.7.1 | 接続時にクライアントリストに追加される | [ ] |
| 8.7.2 | 接続直後に status スナップショットが送信される | [ ] |
| 8.7.3 | 切断時にクライアントリストから除去される | [ ] |
| 8.7.4 | 複数クライアントの同時接続 | [ ] |

### 8.8 ライフサイクル

| # | テスト項目 | 状態 |
|---|-----------|------|
| 8.8.1 | startup で LogBroadcaster が install される | [ ] |
| 8.8.2 | startup で drain タスクが作成される | [ ] |
| 8.8.3 | startup で最新オブジェクトが自動ロードされる | [ ] |
| 8.8.4 | shutdown で LogBroadcaster が uninstall される | [ ] |
| 8.8.5 | shutdown で SAM2 が release される | [ ] |

---

## 9. Backend: Stage Wrappers (`stage_wrappers.py`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 9.1 | 各ラッパーが対応モジュールを遅延インポートする | [ ] |
| 9.2 | `_stage_extract_frames` が正しい引数で stage 関数を呼ぶ | [ ] |
| 9.3 | `_stage_pi3x_inference` が完了後に `cleanup_pytorch_vram` を呼ぶ | [ ] |
| 9.4 | `_stage_diffcd` が register_process / unregister_process を渡す | [ ] |
| 9.5 | 不要なコールバックが del されている | [ ] |

---

## 10. Frontend: Constants (`constants.js`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 10.1 | STAGE_COUNT = 8 | [済] |
| 10.2 | TRANSITION_STAGE_MAX = 7 | [済] |
| 10.3 | MESH_METHOD_DEFAULT = "poisson" | [済] |
| 10.4 | MESH_METHOD_SET に "diffcd" と "poisson" | [済] |
| 10.5 | MESH_REPAIR_THRESHOLD 範囲が有効 | [済] |
| 10.6 | DEFAULT_TAUBIN_NU が負数 | [済] |
| 10.7 | CLASSICAL_PREVIEW_TITLE が非空文字列 | [済] |
| 10.8 | STORAGE_KEYS が全て `clip2mesh:` プレフィックス | [済] |

---

## 11. Frontend: Utils (`utils.js`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 11.1 | `formatTime`: 秒→分:秒 表示 | [済] |
| 11.2 | `normalizeMeshMethod`: 不明値 → "poisson" | [済] |
| 11.3 | `clampMeshRepairThreshold`: 範囲内外のクランプ | [済] |
| 11.4 | `formatMeshRepairThreshold`: 小数点表示 | [済] |
| 11.5 | `parsePositiveInt` / `parseNonNegativeInt` | [済] |
| 11.6 | `parsePositiveFloat` / `parseNonNegativeFloat` | [済] |

---

## 12. Frontend: WsManager (`ws.js`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 12.1 | 接続・切断の状態管理 | [済] |
| 12.2 | `on(type, cb)` / `off(type, cb)` のイベント登録・解除 | [済] |
| 12.3 | メッセージ type に基づくディスパッチ | [済] |
| 12.4 | `'*'` ワイルドカードハンドラ | [済] |
| 12.5 | `_open` / `_close` 内部イベント発火 | [済] |
| 12.6 | 指数バックオフ再接続 (1s → 最大 15s) | [済] |
| 12.7 | JSON パース失敗時のエラーハンドリング | [済] |

---

## 13. Frontend: Router (`router.js`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 13.1 | hash 読み取りでビュー切り替え | [ ] |
| 13.2 | `overview` / `pipeline` のみ有効 | [ ] |
| 13.3 | 不明ハッシュでデフォルト (`overview`) | [ ] |
| 13.4 | `onChange` コールバック発火 | [ ] |
| 13.5 | `body[data-view]` 属性が設定される | [ ] |

---

## 14. Frontend: I18n (`i18n.js`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 14.1 | localStorage に lang なし → "en" デフォルト | [済] |
| 14.2 | localStorage から lang 読み取り | [済] |
| 14.3 | `t(key)`: 英語翻訳 | [済] |
| 14.4 | `t(key)`: 日本語翻訳 (lang="ja") | [済] |
| 14.5 | `t(key)`: 不明キー → キー自体を返す | [済] |
| 14.6 | `t(key, ...params)`: プレースホルダ置換 | [済] |
| 14.7 | `apply()`: `[data-i18n]` 要素の textContent 更新 | [済] |
| 14.8 | `setLang()`: localStorage 更新 + DOM 再翻訳 | [済] |
| 14.9 | `setLang()`: 同一言語は no-op | [済] |

---

## 15. Frontend: PipelineUI (`pipeline.js`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 15.1 | `stageStart`: running クラス設定、タイマー開始 | [済] |
| 15.2 | `stageComplete`: 100% プログレス、タイマー停止 | [済] |
| 15.3 | `stageProgress`: プログレスバー fill 更新 | [済] |
| 15.4 | `stageFailed`: failed クラス設定 | [済] |
| 15.5 | `stageInteractive`: interactive クラス設定 | [済] |
| 15.6 | `updateAll`: ステータススナップショットからの一括更新 | [済] |
| 15.7 | `resetFromStage`: 指定ステージ以降のクリア | [済] |
| 15.8 | `setMeshMethod`: diffcd/poisson の分岐表示切替 | [済] |
| 15.9 | `setMeshMethodEnabled`: method-disabled クラス制御 | [済] |
| 15.10 | `getOverallProgress`: 8 ステージの平均 | [済] |
| 15.11 | コネクタ表示の更新 (`_refreshConnectors`) | [済] |

---

## 16. Frontend: StageController (`stage-controller.js`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 16.1 | `activateStage`: 指定パネルに `active` クラス | [済] |
| 16.2 | `activateStage`: 前のパネルから `active` 除去 | [済] |
| 16.3 | `activateStage`: `stage-activated` CustomEvent 発火 | [済] |
| 16.4 | ピルクリックで activateStage 呼び出し | [済] |
| 16.5 | `setStageState` / `getStageState` | [済] |
| 16.6 | Stage 5 メッシュ方式ピルの切替 | [済] |

---

## 17. Frontend: ConfigPanel (`config-panel.js`)

### 17.1 名前正規化

| # | テスト項目 | 状態 |
|---|-----------|------|
| 17.1.1 | ASCII 名そのまま | [済] |
| 17.1.2 | スラッシュ → ハイフン | [済] |
| 17.1.3 | スペース → ハイフン | [済] |
| 17.1.4 | 特殊文字除去 | [済] |
| 17.1.5 | 連続ハイフン圧縮 | [済] |
| 17.1.6 | 先頭末尾のドット/ハイフン除去 | [済] |
| 17.1.7 | 80 文字切り捨て | [済] |
| 17.1.8 | 空文字列 → 空 | [済] |
| 17.1.9 | 日本語文字保持 | [済] |
| 17.1.10 | 韓国語文字保持 | [済] |
| 17.1.11 | ASCII + 日本語混在 | [済] |

### 17.2 パース関数

| # | テスト項目 | 状態 |
|---|-----------|------|
| 17.2.1 | `_parsePositiveInt`: 正常/0/負/NaN | [済] |
| 17.2.2 | `_parsePositiveFloat`: 正常/0/負 | [済] |
| 17.2.3 | `_parseNonNegativeFloat`: 0 許可/負は fallback | [済] |
| 17.2.4 | `_valuesAlmostEqual`: 同値/微小差/非有限 | [済] |

### 17.3 getConfig

| # | テスト項目 | 状態 |
|---|-----------|------|
| 17.3.1 | video_path キーの存在 | [済] |
| 17.3.2 | object_name キーの存在 | [済] |
| 17.3.3 | 全期待キーの存在 | [済] |
| 17.3.4 | 空入力でデフォルト値使用 | [済] |
| 17.3.5 | texture_view_assign_mode の読み取り | [済] |
| 17.3.6 | texture_quality_boost の読み取り | [済] |

### 17.4 setMeshMethod

| # | テスト項目 | 状態 |
|---|-----------|------|
| 17.4.1 | poisson: classical 表示、diffcd 非表示 | [済] |
| 17.4.2 | diffcd: diffcd 表示、classical 非表示 | [済] |

### 17.5 その他

| # | テスト項目 | 状態 |
|---|-----------|------|
| 17.5.1 | `setRunning(true)` で全入力無効化 | [ ] |
| 17.5.2 | `setRunning(false)` で全入力有効化 | [ ] |
| 17.5.3 | `setActiveStage` でステージ関連セクションのフィルタリング | [ ] |
| 17.5.4 | denoise プリセット選択でパラメータ値の一括設定 | [ ] |
| 17.5.5 | 個別パラメータ変更で preset が "custom" に | [ ] |
| 17.5.6 | `refreshObjects()` でオブジェクト一覧再取得 | [ ] |
| 17.5.7 | ビデオ選択変更で object_name の自動提案 | [ ] |
| 17.5.8 | `onCropScaleChanged` コールバック発火 | [ ] |

---

## 18. Frontend: LogViewer (`log-viewer.js`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 18.1 | 単純行の DOM 追加 | [済] |
| 18.2 | stderr に log-stderr クラス | [済] |
| 18.3 | VRAM/GPU/CUDA テキストに log-vram クラス | [済] |
| 18.4 | ANSI エスケープシーケンス除去 | [済] |
| 18.5 | 改行で複数エントリに分割 | [済] |
| 18.6 | `\r` がプログレスヒントとして扱われる | [済] |
| 18.7 | 末尾改行なしテキストのバッファリング | [済] |
| 18.8 | 次チャンクの改行でバッファフラッシュ | [済] |
| 18.9 | "ETA" / "it/s" / パーセント表示のプログレス検出 | [済] |
| 18.10 | プログレスコンパクション ([progress xN]) | [済] |
| 18.11 | アクティブステージのみ表示 | [済] |
| 18.12 | ステージ切替で正しいエントリ再描画 | [済] |
| 18.13 | `setMaxLines`: 有効値受け付け | [済] |
| 18.14 | `setMaxLines`: 100 未満拒否 | [済] |
| 18.15 | `setMaxLines`: 50000 上限 | [済] |
| 18.16 | `setMaxLines`: NaN 拒否 | [済] |
| 18.17 | maxLines 超過時のトリミング | [済] |
| 18.18 | `clear()` で全エントリ・DOM クリア | [済] |
| 18.19 | `_normalizeStage`: NaN → 1, float 丸め, 範囲外 → 1 | [済] |

---

## 19. Frontend: CheckpointPanel (`checkpoint-panel.js`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 19.1 | clampStage: NaN → 1, 範囲外クランプ, float 丸め | [済] |
| 19.2 | normalizeStatus: 不明 → pending | [済] |
| 19.3 | normalizeMeshMethod: 不明 → poisson | [済] |
| 19.4 | isTransitionConfirmationDetail: "continue to" / "stage N complete" | [済] |
| 19.5 | checkpoint_id によるインデックス解決 | [済] |
| 19.6 | detail regex によるフォールバックインデックス解決 | [済] |
| 19.7 | "complete" detail → maxIndex | [済] |
| 19.8 | 一致なし → 前回インデックス維持 | [済] |
| 19.9 | complete/pending/running/failed/interactive のチェックポイント状態遷移 | [済] |
| 19.10 | Stage 5 poisson/diffcd テンプレート切替 | [済] |
| 19.11 | "waiting for next-stage confirmation" → 全 complete | [済] |
| 19.12 | reset(1): 全 pending / reset(3): 1–2 complete, 3–8 pending | [済] |
| 19.13 | lifecycle: start → progress → complete / failed | [済] |
| 19.14 | `applyStatusSnapshot`: ステータスからの一括適用 | [済] |
| 19.15 | DOM リストアイテムの更新 | [済] |
| 19.16 | Stage 7 ground-plane detail でのチェックポイント進行 | [済] |

---

## 20. Frontend: TaskConfirmController (`task-confirm-controller.js`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 20.1 | 初期状態で全バー idle | [済] |
| 20.2 | `setVisibleStage` で指定バーのみ表示 | [済] |
| 20.3 | `setWaiting` でボタン有効化 + メッセージ表示 | [済] |
| 20.4 | `confirmNextStage` で POST 送信 | [済] |
| 20.5 | API 成功後に sending 状態 | [済] |
| 20.6 | API 失敗後に waiting に戻る | [済] |
| 20.7 | `setConfirmed` でボタン無効化 | [済] |
| 20.8 | `setIdle` でボタン無効化 + メッセージクリア | [済] |
| 20.9 | `syncFromStatus` でスナップショットからの同期 | [済] |
| 20.10 | waitingStage / waitingToStage getter | [済] |
| 20.11 | Stage 8 (final) ではボタン非表示 | [済] |

---

## 21. Frontend: SAM2Canvas (`sam2-canvas.js`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 21.1 | `activate` でフレーム画像ロード | [済] |
| 21.2 | 左クリックで positive ポイント (label=1) | [済] |
| 21.3 | 右クリックで negative ポイント (label=0) | [済] |
| 21.4 | クリックで `POST /api/sam2/click` 送信 | [済] |
| 21.5 | 座標の正規化 (0–1 範囲) | [済] |
| 21.6 | `undo` で `POST /api/sam2/undo` 送信 | [済] |
| 21.7 | `clear` で `POST /api/sam2/clear` 送信 | [済] |
| 21.8 | `confirm` で `POST /api/sam2/confirm` 送信 | [済] |
| 21.9 | `deactivate` でキャンバス非アクティブ化 | [済] |
| 21.10 | ローディング中はクリック無効 | [済] |
| 21.11 | `enterGroundPhase` でモード切替 | [済] |
| 21.12 | `exitGroundPhase` でモード復帰 | [済] |
| 21.13 | skip-ground ボタンで `POST /api/sam2/skip-ground` | [済] |
| 21.14 | positive/negative カウント表示更新 | [済] |

---

## 22. Frontend: SAM2Verification (`sam2-verification.js`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 22.1 | `show` で verification ストリップ表示 | [済] |
| 22.2 | フレームインデックスの均等分配 (`_pickIndices`) | [済] |
| 22.3 | approve ボタンで `POST /api/sam2/approve` | [済] |
| 22.4 | redo ボタンで `POST /api/sam2/redo` | [済] |
| 22.5 | hide 後に DOM がクリアされる | [済] |
| 22.6 | ground overlay 画像のロード (hasGround=true) | [済] |

---

## 23. Frontend: CameraOverlay (`camera-overlay.js`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 23.1 | 初期状態で _group=null | [済] |
| 23.2 | `remove`: null group で例外なし | [済] |
| 23.3 | `setVisible`: null group で例外なし | [済] |
| 23.4 | `applyOffset`: null group で例外なし | [済] |
| 23.5 | `create` でフラスタム作成 | [ ] |
| 23.6 | `setVisible(false)` でグループ非表示 | [ ] |

---

## 24. Frontend: PreviewPanel (`preview.js`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 24.1 | `activateStage` で renderer を対応コンテナに移動 | [済] |
| 24.2 | `loadGallery` でフレーム画像のサムネイル生成 | [済] |
| 24.3 | `reset` で全ステージシーンクリア | [済] |
| 24.4 | `clearFromStage` で指定ステージ以降クリア | [済] |
| 24.5 | `applyTheme` でバックグラウンド色変更 | [済] |
| 24.6 | `loadStageResult` で PLY/OBJ ファイルの判定 | [済] |
| 24.7 | `beginMeshRepairSelection` でループオーバーレイ表示 | [済] |
| 24.8 | `setMeshRepairThreshold` でループの表示/非表示切替 | [済] |
| 24.9 | `getMeshRepairSelectedLoopIds` で選択 ID 返却 | [済] |
| 24.10 | `setMeshRepairConfirmed` で色変更 | [済] |
| 24.11 | `clearMeshRepairSelection` でオーバーレイクリア | [済] |
| 24.12 | `showCropBbox` で OBB ワイヤーフレーム表示 | [済] |
| 24.13 | `updateCropBbox` でスケール更新 | [済] |
| 24.14 | `clearCropBbox` で OBB 除去 | [済] |
| 24.15 | シーンフリップ判定 (`_shouldApplySceneFlipX`) | [済] |
| 24.16 | Pi3X ポイントクラウドのロードとカメラポーズ表示 | [ ] |
| 24.17 | OBJ + MTL テクスチャ付きメッシュのロード | [ ] |

---

## 25. Frontend: StatusHydrator (`status-hydrator.js`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 25.1 | `applySnapshot` で全モジュールに伝播 | [済] |
| 25.2 | 同一 statusKey で重複 hydration スキップ | [済] |
| 25.3 | `force=true` で重複チェックバイパス | [済] |
| 25.4 | running 状態のスナップショット適用 | [済] |
| 25.5 | idle 状態のスナップショット適用 | [済] |
| 25.6 | `hydrateOutputs`: 完了ステージのプレビューロード | [済] |
| 25.7 | mesh_method のスナップショットからの反映 | [済] |

---

## 26. Frontend: SettingsPanel (`settings-panel.js`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 26.1 | デフォルトテーマ "light" | [済] |
| 26.2 | localStorage からテーマ読み取り | [済] |
| 26.3 | テーマ切替で `data-theme` 属性設定 | [済] |
| 26.4 | 言語切替で localStorage 更新 | [済] |
| 26.5 | open/close トグル | [済] |
| 26.6 | Escape キーで close | [済] |
| 26.7 | auto-scroll チェックボックスの localStorage 永続化 | [済] |
| 26.8 | max-lines 入力の localStorage 永続化 | [済] |
| 26.9 | auto-accept チェックボックスの localStorage 永続化 | [済] |
| 26.10 | `onThemeChanged` コールバック発火 | [済] |
| 26.11 | `onLangChanged` コールバック発火 | [済] |
| 26.12 | `onLogSettingsChanged` コールバック発火 | [済] |

---

## 27. Frontend: PipelineStatus (`pipeline-status.js`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 27.1 | `getStageInfo`: null/missing → null, 正常 → stage info | [済] |
| 27.2 | `isStageDone`: complete/progress>=100 → true | [済] |
| 27.3 | `isStageAvailable`: complete/interactive/progress>0 → true | [済] |
| 27.4 | `resolvePreferredStage`: null → 1, current_stage 有効なら採用 | [済] |
| 27.5 | `buildStatusKey`: 変更検出フィンガープリント | [済] |
| 27.6 | `isTransitionConfirmed`: 確認済み判定 | [済] |
| 27.7 | `resolveTransitionTarget`: 遷移先ステージ解決 | [済] |
| 27.8 | デフォルトメッセージテキスト生成関数 | [済] |

---

## 28. Frontend: MeshPostController (`mesh-post-controller.js`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 28.1 | init でツールバー非表示、コントロール無効化 | [済] |
| 28.2 | `setToolbarVisible` の表示/非表示切替 | [済] |
| 28.3 | `setEnabled` で入力・ボタンの有効/無効化 | [済] |
| 28.4 | `setStatus` でテキスト・トーンクラス設定 | [済] |
| 28.5 | `canRun`: 非実行中 → true, 他ステージ実行中 → false, Stage 5 待ち → true | [済] |
| 28.6 | `syncFromStatus`: Stage 5 完了 + object_name あり → 表示 | [済] |
| 28.7 | `syncFromStatus`: Stage 5 未完了 → 非表示 | [済] |
| 28.8 | `syncFromStatus`: inFlight 中は disabled 維持 | [済] |
| 28.9 | `applyPostprocess`: POST ボディのフォーマット | [済] |
| 28.10 | `applyPostprocess`: iterations クランプ (0–100) | [済] |
| 28.11 | reset ボタンの動作 | [済] |

---

## 29. Frontend: MeshRepairController (`mesh-repair-controller.js`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 29.1 | init でツールバー非表示 | [済] |
| 29.2 | `activateFromApi`: 候補取得 → プレビューに描画 | [済] |
| 29.3 | threshold スライダーで `preview.setMeshRepairThreshold` 呼び出し | [済] |
| 29.4 | confirm ボタンで `POST /api/mesh-repair/confirm` 送信 | [済] |
| 29.5 | confirm 成功後 `preview.setMeshRepairConfirmed()` 呼び出し | [済] |
| 29.6 | `syncFromStatus`: Stage 7 interactive で再構築 | [済] |
| 29.7 | カウントラベル (selected/candidates/visible/threshold) | [済] |
| 29.8 | clear ボタンで選択リセット | [済] |
| 29.9 | `deactivate` でツールバー非表示 + プレビュークリア | [済] |

---

## 30. Frontend: Overview (`overview.js`)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 30.1 | `refresh` で `GET /api/pipeline/objects` から一覧取得 | [ ] |
| 30.2 | カード表示 (サムネイル、名前、ステージドット) | [ ] |
| 30.3 | カードクリックで `onOpenObject` コールバック | [ ] |
| 30.4 | 新規パイプラインボタンで `onNewPipeline` コールバック | [ ] |
| 30.5 | `markStale` → `refreshIfStale` で再取得 | [ ] |
| 30.6 | `setActiveObject` でアクティブ表示 | [ ] |
| 30.7 | オブジェクトなし時の空プレースホルダ表示 | [ ] |

---

## 31. Frontend: app.js (統合ワイヤリング)

| # | テスト項目 | 状態 |
|---|-----------|------|
| 31.1 | WS `status` メッセージで statusHydrator.applySnapshot 呼び出し | [ ] |
| 31.2 | WS `stage_start` でステージ切替 + UI 更新 | [ ] |
| 31.3 | WS `stage_complete` でプレビューロード | [ ] |
| 31.4 | WS `sam2_ready` で SAM2 キャンバスアクティベート | [ ] |
| 31.5 | WS `sam2_verification_ready` で verification ストリップ表示 | [ ] |
| 31.6 | WS `mesh_repair_ready` でメッシュ修復 UI アクティベート | [ ] |
| 31.7 | WS `pipeline_complete` で完了 UI 表示 | [ ] |
| 31.8 | WS `pipeline_error` でエラー UI 表示 | [ ] |
| 31.9 | WS `next_stage_confirmation_required` で confirm バー表示 | [ ] |
| 31.10 | VRAM ポーリング (5 秒間隔) | [ ] |
| 31.11 | overview ↔ pipeline ビュー切替 | [ ] |
| 31.12 | オブジェクト選択で `POST /api/pipeline/load-object` | [ ] |
| 31.13 | Start ボタンで `POST /api/pipeline/start` | [ ] |
| 31.14 | Cancel ボタンで `POST /api/pipeline/cancel` | [ ] |
| 31.15 | `_objectLoadRequestId` によるスタル応答の排除 | [ ] |

---

## 集計

| カテゴリ | 済 | 未実装 | 合計 |
|---------|---:|------:|-----:|
| Backend: State | 39 | 5 | 44 |
| Backend: Configuration | 22 | 3 | 25 |
| Backend: Checkpoints | 23 | 0 | 23 |
| Backend: Log Capture | 16 | 0 | 16 |
| Backend: SAM2 Service | 18 | 8 | 26 |
| Backend: Pipeline Runner | 20 | 15 | 35 |
| Backend: Object Store | 17 | 6 | 23 |
| Backend: App Endpoints | 14 | 23 | 37 |
| Backend: Stage Wrappers | 0 | 5 | 5 |
| Frontend: Constants | 8 | 0 | 8 |
| Frontend: Utils | 6 | 0 | 6 |
| Frontend: WsManager | 7 | 0 | 7 |
| Frontend: Router | 0 | 5 | 5 |
| Frontend: I18n | 9 | 0 | 9 |
| Frontend: PipelineUI | 11 | 0 | 11 |
| Frontend: StageController | 6 | 0 | 6 |
| Frontend: ConfigPanel | 18 | 8 | 26 |
| Frontend: LogViewer | 19 | 0 | 19 |
| Frontend: CheckpointPanel | 16 | 0 | 16 |
| Frontend: TaskConfirmController | 11 | 0 | 11 |
| Frontend: SAM2Canvas | 14 | 0 | 14 |
| Frontend: SAM2Verification | 6 | 0 | 6 |
| Frontend: CameraOverlay | 4 | 2 | 6 |
| Frontend: PreviewPanel | 15 | 2 | 17 |
| Frontend: StatusHydrator | 7 | 0 | 7 |
| Frontend: SettingsPanel | 12 | 0 | 12 |
| Frontend: PipelineStatus | 8 | 0 | 8 |
| Frontend: MeshPostController | 11 | 0 | 11 |
| Frontend: MeshRepairController | 9 | 0 | 9 |
| Frontend: Overview | 0 | 7 | 7 |
| Frontend: app.js 統合 | 0 | 15 | 15 |
| **合計** | **410** | **104** | **514** |

---

## 優先度別の未実装テスト分類

### P0: セキュリティ / データ整合性 ✅ 全件実装済み

- ~~8.5.2 `GET /api/preview/file/{path}`: パストラバーサル防止~~
- ~~8.5.4 `GET /api/preview/object-file/{name}/{path}`: パス脱出 → 403~~
- ~~8.1.20 `GET /api/pipeline/video-info`: INPUT_DIR 外パス → 403~~

### P1: コア API エンドポイント (高優先度)

- 8.1.7–8.1.21 Pipeline API の未実装テスト群
- 8.2.2–8.2.15 SAM2 API の未実装テスト群
- 8.4.1–8.4.5 Mesh Post-Process API
- 8.7.1–8.7.4 WebSocket 接続管理

### P2: パイプライン統合フロー (中優先度)

- 6.8.1–6.8.11 run_pipeline 統合テスト
- 6.9.1–6.9.4 _run_stage テスト
- 31.1–31.15 フロントエンド統合ワイヤリング

### P3: エッジケース補完 (低優先度)

- 残りの State / Configuration / SAM2 Service / Object Store の個別テスト
- Router, Overview, CameraOverlay の未実装テスト
