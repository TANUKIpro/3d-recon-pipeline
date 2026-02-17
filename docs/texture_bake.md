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

既定値の定義元: `scripts/config_defaults.py`

## パフォーマンス最適化

テクスチャベイク処理には以下の 6 つの最適化が適用されている。

- **A. メモリ適応型 LRU キャッシュ**: フレーム画像とマスクを `_FrameCache` でキャッシュ。利用可能メモリからキャッシュ容量を自動算出し、フレーム 70% / マスク 30% の予算比率で確保する。
- **B. 深度ラスタライズ前処理のベクトル化**: 面ごとの UV 座標・バウンディングボックス・重心分母を NumPy でバッチ計算し、ラスタライズループ前にフィルタリングする。
- **C. 深度バッファ LRU キャッシュ**: 視点ごとの深度バッファを `@lru_cache` でキャッシュし、同一視点の再計算を回避する。
- **D. 内在パラメータ推定の並列化**: `ThreadPoolExecutor` で複数フレームの色スコア評価を並列実行する。
- **E. テクセル UV 座標の事前計算**: 三角形ごとの UV 座標を `np.stack` で事前計算し、`fillConvexPoly` で再利用する。
- **F. OBJ バッファ書き出し**: OBJ ファイルを文字列リストに蓄積してから一括書き込みする。

### 内部メモリ管理パラメータ

| 名前 | 既定値 | 説明 |
|---|---:|---|
| `_TEXTURE_CACHE_SAFETY_MB` | `1024` | キャッシュ割り当て時のメモリ安全マージン (MB) |
| `_TEXTURE_FRAME_BUDGET_RATIO` | `0.7` | フレームキャッシュ予算比率 |
| `_TEXTURE_MASK_BUDGET_RATIO` | `0.3` | マスクキャッシュ予算比率 |
| `_TEXTURE_MEM_FALLBACK_MB` | `4096` | メモリ情報取得失敗時のフォールバック容量 (MB) |

## 失敗時の典型原因

- pose と frame/mask の index 不整合
- `camera_poses.json` に pose が無い
- 入力メッシュが空

## 参考文献

- xatlas repository: <https://github.com/jpcy/xatlas>
- Nelder-Mead (SciPy optimize.minimize): <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html>
