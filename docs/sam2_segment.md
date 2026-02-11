# sam2_segment

## 対象タスク

- Stage 3: SAM2 セグメンテーション
- 実装:
  - `scripts/stage_sam2_ui.py` (SAM2 セッションと推論)
  - `scripts/dashboard/sam2_service.py` (REST/API 向けラッパ)
  - `scripts/dashboard/pipeline_runner.py` (Stage 3 制御)

## 概要

ユーザーのクリック操作で対象物体を指定し、SAM2 で全フレームへマスクを伝播する。  
完了後に Stage 2 の `pi3x_cache.npz` へマスクを適用し、対象物体点群 `object.ply` を生成する。

## 入出力関係

前段入力:

- `<output_dir>/frames/*.jpg`
- `<output_dir>/pi3x_cache.npz` (Stage 2 生成)

主出力:

- `<output_dir>/masks/*.png`
- `<output_dir>/object.ply` (SAM2マスク適用後の点群)

後段利用:

- Stage 4 (`denoise_point_cloud`) が `object.ply` を入力
- Stage 8 (`texture_bake`) が `masks/` を利用

## 詳細フロー

1. SAM2 初期化:
   - フレーム読み込み
   - 推論状態 (`init_state`) 構築
2. クリック操作:
   - 左クリック=positive (`label=1`)
   - 右クリック=negative (`label=0`)
3. 各クリックごとに frame0 で即時再推論し、オーバーレイ表示更新。
4. `Confirm & Propagate` で全フレームへマスク伝播して `masks/*.png` 保存。
5. Dashboard では検証UIで `Approve/Redo` 可能。
6. 承認後、`apply_sam2_masks` を実行して `object.ply` を生成。

## アルゴリズム要点

- インタラクティブ点ベースセグメンテーション (SAM2 video predictor)
- クリック点は正規化座標で管理し、実画像座標に変換して推論
- 伝播後に Pi3X の conf+edge フィルタ結果へ論理積を取る

## パラメータ

| 名前 | 既定値 | 説明 |
|---|---:|---|
| `SAM2_MODEL` | `large` | SAM2モデル種別の設定値 |
| `model_type` 引数 | `None` | `None` の場合 `SAM2_MODEL` を参照 |

実装上の注意:

- `SAM2Session._load_model()` は現状 `SAM2_MODEL` を参照せず `large` を固定ロードする。
- そのため Dashboard で `tiny/small/base` を選択しても、現在の実体は `large` になる。

## 失敗時の典型原因

- `No JPEG frames ...`: Stage 1/2 の成果物不足
- `No click points to propagate`: クリック未指定で確定した場合
- マスク欠損: `apply_sam2_masks()` は欠損マスクをゼロ扱いで継続

## 参考文献

- SAM2 repository: <https://github.com/facebookresearch/sam2>
- Segment Anything 2 (Meta AI): <https://arxiv.org/abs/2408.00714>
