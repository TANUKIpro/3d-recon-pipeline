# pi3x_reconstruct

## 対象タスク

- Stage 2: Pi3X 3D 再構成
- 実装: `scripts/stage_pi3x_reconstruct.py`
- 呼び出し元: `scripts/dashboard/pipeline_runner.py`, `scripts/pipeline.py`

## 概要

抽出済みフレーム群から Pi3X で点群とカメラ姿勢を推定し、  
まずは `信頼度 + 深度エッジ` まで適用した点群を保存する。  
SAM2 マスク適用は Stage 3 で `pi3x_cache.npz` を使って後適用する構成。

## 入出力関係

前段入力:

- `<output_dir>/frames/*.jpg`

主出力 (Stage 2 終了時点):

- `<output_dir>/object_full.ply` (conf + edge 済み点群)
- `<output_dir>/camera_poses.json` (4x4 pose + frame index + alignment metadata)
- `<output_dir>/pi3x_cache.npz` (Stage 3 へ渡す中間キャッシュ)

後段利用:

- Stage 3 が `pi3x_cache.npz` と `masks/*.png` を合成して `object.ply` を生成
- Stage 6 が `camera_poses.json` を利用

## 詳細フロー

1. 入力フレーム列挙、`max_frames` に応じて等間隔サンプリング (`linspace`)。
2. `pixel_limit` を超える場合のみリサイズしてテンソル化。
3. Pi3X をロードし、`encode -> decode -> forward_head` の分割実行で VRAM を節約。
4. OOM 時は段階フォールバック:
   - フレーム数削減 (`x0.8`, 最低 12)
   - `pixel_limit` 削減 (`x0.7`, 最低 50,000)
   - さらに必要ならチャンク推論 + 重複区間 Procrustes 整合
5. 任意でカメラ軌道平面を基準面へ整列 (`ALIGN_CAMERA_PLANE`)。
6. フィルタ適用:
   - `sigmoid(conf) > CONFIDENCE_THRESHOLD`
   - `~depth_edge(local_points[...,2], rtol=EDGE_RTOL)`
7. `object_full.ply`, `camera_poses.json`, `pi3x_cache.npz` を保存。

## Stage 3 との境界 (apply_sam2_masks)

同ファイル内の `apply_sam2_masks()` が Stage 3 で使用される。

1. `pi3x_cache.npz` 読み込み
2. `frame_indices` に対応する `masks/*.png` をロード
3. `conf_edge_mask & sam2_mask` を適用
4. `<output_dir>/object.ply` を保存

## アルゴリズム要点

- メモリ効率推論: モデルサブモジュールを都度 CPU オフロード。
- OOM フォールバック: 品質劣化が小さい順に対処 (フレーム数 -> 解像度 -> チャンク化)。
- チャンク結合: 重複フレームのカメラ中心で Procrustes 変換を推定し、後続チャンクを整列。
- ワールド整列: カメラ軌道の主平面推定により座標系を正規化。

## パラメータ

| 名前 | 既定値 | 説明 |
|---|---:|---|
| `PIXEL_LIMIT` | `255000` | フレーム1枚あたりの最大画素数。超える場合のみ縮小 |
| `MAX_FRAMES` | `50` | Pi3X 推論に使うフレーム上限 |
| `CONFIDENCE_THRESHOLD` | `0.2` | 信頼度フィルタ閾値 |
| `EDGE_RTOL` | `0.03` | 深度エッジ除去の相対閾値 |
| `ALIGN_CAMERA_PLANE` | `1` | カメラ軌道平面に基づく座標整列のON/OFF |
| `PI3X_VRAM_TARGET_UTILIZATION` | `0.95` | 事前推定 (dashboard 表示) のVRAM目標使用率 |
| `PI3X_ESTIMATED_MODEL_MB` | `5500` | 事前推定に使うモデルVRAM見積 |
| `PI3X_RUNTIME_OVERHEAD_MB` | `1200` | 事前推定に使う実行オーバーヘッド |
| `PI3X_FRAME_PIXELS_PER_MB` | `800` | 画素量->VRAM換算の近似係数 |
| `pi3x_frame_target` (Dashboard設定) | `max_frames` 由来 | Stage 2 で実際に使うフレーム数目標 |

## 失敗時の典型原因

- `Need at least 2 frames for reconstruction.`: 入力フレーム不足
- `CUDA OOM ...`: フォールバックでも収まらないケース
- `Failed to read frame`: 破損画像またはパス不整合

## 参考文献

- Pi3X repository: <https://github.com/yyfz/Pi3>
- Procrustes alignment (Umeyama 1991): <https://ieeexplore.ieee.org/document/88573>
