# post_texture_cleanup

## 対象タスク

- Stage 6: Post-texture Contact Cleanup
- 実装:
  - `scripts/stage_post_texture_contact_cleanup.py` (メインロジック)
  - `scripts/ground_plane_extraction.py` (接地平面推定)
  - `scripts/repair/` (境界抽出、三角形分割、接地平面クリッピング等のヘルパー)
- 呼び出し元: `scripts/dashboard/pipeline_runner.py`, `scripts/pipeline.py`

## 概要

Stage 5 で生成されたテクスチャ付き OBJ に対して、接地面のアーティファクト (地面との接触部分に発生する余分なメッシュ) を検出・除去する。ダッシュボードでは提案をプレビューし、ユーザーが Apply/Skip を選択する。

## 入出力関係

前段入力:

- `<output_dir>/textured_mesh.obj` (Stage 5 出力)
- `<output_dir>/camera_poses.json` (Stage 2 出力)
- `<output_dir>/intrinsics.json` (Stage 2 出力)
- `<output_dir>/masks/*.png` (Stage 3 出力)
- `<output_dir>/masks_ground/*.png` (Stage 3 出力, 任意)
- `<output_dir>/ground_plane.json` (自動生成)
- `<output_dir>/object_mesh.ply` (Stage 4 出力, 接地平面推定用)

主出力:

- `<output_dir>/<object_name>/textured_mesh_cleaned.obj` — 最終成果物
- `<output_dir>/<object_name>/textured_mesh_cleaned.mtl`
- `<output_dir>/<object_name>/texture.png` — Stage 5 のテクスチャコピー
- `<output_dir>/<object_name>/texture_cap.png` — 接地キャップテクスチャ (apply 時)

中間ファイル:

- `<output_dir>/post_texture_contact_cleanup/proposal.json` — 提案メタデータ
- `<output_dir>/post_texture_contact_cleanup/proposal_removed_region.ply` — 除去対象領域のプレビュー

## 詳細フロー

### 1. 接地平面推定

- `ground_plane.json` が存在しない場合、`scripts/ground_plane_extraction.py` を使用して自動推定。
- SAM2 の ground masks + カメラ投影 + メッシュ形状から接地平面を決定。

### 2. クリーンアップ提案生成

- テクスチャ付き OBJ メッシュを読み込み、face ごとに以下を評価:
  - ground mask への投影一致率
  - object mask への投影一致率
  - 接地平面からの距離
  - 近傍 face の除去状態 (伝播)
- 複数パスで除去候補を収束:
  - ground mask 投影 > 閾値の face を除去候補
  - object mask 投影 < 閾値の face を除去候補
  - 接地平面近傍の突出 face を除去候補
  - 浮遊アイランドの除去
  - 収束ループ (最大3回)
- 提案をJSON + PLY プレビューとして保存。

### 3. レビュー判定

- **ダッシュボード**: 提案の3Dプレビューを表示し、ユーザーが Apply/Skip を選択。
- **CLI**: `--post-texture-cleanup-selection-json` で JSON ファイルを指定 (`{"decision": "apply"}` / `{"decision": "skip"}`)。未指定時はスキップ。

### 4a. Apply の場合

- 接地平面でメッシュをクリッピング (平面の少し上側で切断)。
- 底面の穴を三角形分割 (Ear clipping) でキャップ化。
- スカートキャップ (底面からの垂直延長) を生成して自然な見た目に。
- キャップ部分のテクスチャを `texture_cap.png` として別途生成 (または `texture.png` に統合)。
- 最終成果物を `<object_name>/` サブフォルダに出力。

### 4b. Skip の場合

- Stage 5 の OBJ/MTL/texture.png を `<object_name>/` サブフォルダにそのままコピー。

## パラメータ

| 名前 | 既定値 | 説明 |
|---|---:|---|
| `POST_TEXTURE_CLEANUP_ENABLED` | `true` | ステージの有効/無効 |
| `CLEANUP_LOWER_HALF_THRESHOLD` | `0.2` | メッシュ下半分での除去判定閾値 |
| `REPAIR_MAX_DIAMETER_RATIO` | `0.46` | 境界ループの最大直径比 (バウンディングボックス比) |
| `REPAIR_Y_BAND_RATIO` | `0.06` | 接地 Y 帯域の比率 |
| `REPAIR_SMOOTH_ITERS` | `3` | 局所スムージング反復回数 |

### 内部しきい値

| 名前 | 値 | 説明 |
|---|---:|---|
| `_MASK_GROUND_REMOVAL_THRESHOLD` | `0.5` | ground mask 投影比率の除去閾値 |
| `_MASK_OBJECT_PRESERVATION_THRESHOLD` | `0.2` | object mask 投影比率の保存閾値 |
| `_COMPONENT_MIN_FACE_RATIO` | `0.02` | 最小コンポーネント face 比率 |
| `_CONVERGENCE_MAX_ITERATIONS` | `3` | 収束ループ最大回数 |
| `_ISLAND_FACE_RATIO` | `0.1` | 浮遊アイランド判定の face 比率 |
| `_GROUND_PLANE_RAISE_RATIO` | `0.015` | 接地平面の上方オフセット比率 |

既定値の定義元: `scripts/config_defaults.py`, `scripts/stage_post_texture_contact_cleanup.py`

## 出力構造

```
<output_dir>/
  <object_name>/                    ← 最終成果物フォルダ (自己完結型)
    textured_mesh_cleaned.obj
    textured_mesh_cleaned.mtl
    texture.png
    texture_cap.png                 ← apply 時のみ
  post_texture_contact_cleanup/
    proposal.json
    proposal_removed_region.ply
```

## 失敗時の典型原因

- 接地平面が推定できない: ground masks が存在しない場合、ステージはスキップされる
- 提案生成で除去対象が0件: requires_review=false となり、自動的にスキップ扱い

## 参考文献

- Ear clipping 三角形分割: <https://en.wikipedia.org/wiki/Polygon_triangulation#Ear_clipping_method>
