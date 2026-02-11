# texture_bake

## 対象タスク

- Stage 8: テクスチャベイク
- 実装: `scripts/stage_texture_bake.py`

## 概要

メッシュ・カメラ姿勢・入力フレーム・マスクから、  
`textured_mesh.obj/.mtl` と `texture.png` を生成する。

## 入出力関係

前段入力:

- `<output_dir>/object_mesh_repaired.ply`
- `<output_dir>/camera_poses.json`
- `<output_dir>/frames/*.jpg`
- `<output_dir>/masks/*.png`

主出力:

- `<output_dir>/intrinsics.json`
- `<output_dir>/texture.png`
- `<output_dir>/textured_mesh.mtl`
- `<output_dir>/textured_mesh.obj`

## 詳細フロー

1. メッシュと pose 読み込み、対応フレーム index を解決。
2. 内部パラメータ推定:
   - FOV グリッドサーチ
   - Nelder-Mead で `(fx, fy, cx, cy)` 最適化
3. `xatlas.parametrize` で UV 展開。
4. UVチャート（連結面）ごとに最適視点を選定。
   - スコア: 法線角度 + 距離 + 可視性（簡易Zテスト）+ SAM2マスク
   - チャート単位で primary 視点を貼り付け
   - 未充填テクセルは次点視点を使って二次候補再探索
5. UV seam 周辺を反復補間して隙間埋め。
6. 必要なら supersample -> downsample、sharpen を適用して PNG 書き出し。
7. OBJ/MTL を生成。

## アルゴリズム要点

- 視点選定:
  - `normal・view_dir` の余弦に指数 (`TEXTURE_ANGLE_EXP`) を適用
  - 距離減衰 (`TEXTURE_DIST_POW`) を適用
  - 可視性はビューごとの深度ラスタライズで判定
- 貼り付け:
  - UVチャート単位で最適視点を1つ選ぶ
  - 欠損は二次候補で再探索してから seam padding
- マスク考慮:
  - 投影点が SAM2 マスク内にあるサンプルのみ採用

## パラメータ

| 名前 | 既定値 | 説明 |
|---|---:|---|
| `TEXTURE_SIZE` | `0` | 最終テクスチャ解像度。`0` 以下は `round(sqrt(video_width * video_height))` の正方形を自動適用 |
| `TEXTURE_DEVICE` | `cuda` | 実行要求ヒント (`cuda` / `auto` / `cpu`)。現行のチャート選定処理は CPU で実行 |
| `TEXTURE_OVERSAMPLE` | `2` | 内部解像度倍率 |
| `TEXTURE_MIN_COS` | `0.2` | 面法線と視線方向の最小余弦 |
| `TEXTURE_ANGLE_EXP` | `2.0` | 角度重み指数 |
| `TEXTURE_DIST_POW` | `1.0` | 距離減衰指数 |
| `TEXTURE_SHARPEN` | `0.15` | 最終アンシャープ量 |

## 失敗時の典型原因

- pose と frame/mask の index 不整合
- `camera_poses.json` に pose が無い
- 入力メッシュが空

## 参考文献

- xatlas repository: <https://github.com/jpcy/xatlas>
- Nelder-Mead (SciPy optimize.minimize): <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html>
