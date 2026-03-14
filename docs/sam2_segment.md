# sam2_segment

## 対象タスク

- Stage 3: SAM2 セグメンテーション
- 実装:
  - `scripts/stage_sam2_ui.py` (SAM2 セッションと推論)
  - `scripts/dashboard/sam2_service.py` (REST/API 向けラッパ)
  - `scripts/dashboard/pipeline_runner.py` (Stage 3 制御)

## 概要

ユーザーのクリック操作で対象物体と ground/contact surface を指定し、SAM2 で全フレームへマスクを伝播する。
保存時に `object_raw AND NOT ground_raw` で canonical final mask を合成し、後段ステージへ渡す。

## 入出力関係

前段入力:

- `<output_dir>/frames/*.jpg`

主出力:

- `<output_dir>/masks/*.png` (canonical final mask)
- `<output_dir>/masks_object_raw/*.png`
- `<output_dir>/masks_ground/*.png` (ground 指定時のみ)

後段利用:

- Stage 4 (`gs2mesh_reconstruct`) が `masks/` を TSDF 用 `left_mask.npy` へ変換
- Stage 5 (`texture_bake`) が `masks/` を利用

## 詳細フロー

1. SAM2 初期化:
   - フレーム読み込み
   - 推論状態 (`init_state`) 構築
2. クリック操作:
   - 左クリック=positive (`label=1`)
   - 右クリック=negative (`label=0`)
3. 各クリックごとに frame0 で即時再推論し、オーバーレイ表示更新。
4. Object 確定後、必要なら ground/contact surface を別フェーズで指定する。
5. 伝播時に raw object / raw ground を保存し、`masks/` に final mask を保存する。
6. Dashboard では検証UIで `Approve/Redo` 可能。

## アルゴリズム要点

- インタラクティブ点ベースセグメンテーション (SAM2 video predictor)
- クリック点は正規化座標で管理し、実画像座標に変換して推論
- final mask は `object_raw AND NOT ground_raw`
- 後段は常に `masks/` を canonical source of truth として扱う

## パラメータ

| 名前 | 既定値 | 説明 |
|---|---:|---|
| `SAM2_MODEL` | `large` | SAM2モデル種別の設定値 |
| `model_type` 引数 | `None` | `None` の場合 `SAM2_MODEL` を参照 |

既定値の定義元: `scripts/config_defaults.py`

実装上の注意:

- `SAM2Session._load_model()` は現状 `SAM2_MODEL` を参照せず `large` を固定ロードする。
- そのため Dashboard で `tiny/small/base` を選択しても、現在の実体は `large` になる。

## 失敗時の典型原因

- `No JPEG frames ...`: Stage 1/2 の成果物不足
- `No click points to propagate`: クリック未指定で確定した場合
- `camera_data.json` と `masks/` の frame index が対応しない: Stage 4 の TSDF マスク生成が停止する

## 参考文献

- SAM2 repository: <https://github.com/facebookresearch/sam2>
- Segment Anything 2 (Meta AI): <https://arxiv.org/abs/2408.00714>
