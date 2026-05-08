# texture_bake

## 対象タスク

- Stage 5: テクスチャベイク
- 実装: `scripts/stage_texture_bake.py`, `scripts/texture/` サブパッケージ

## 概要

メッシュ・カメラ姿勢・入力フレーム・マスクから、  
`textured_mesh.obj/.mtl` と `texture.png` を生成する。

## 入出力関係

前段入力:

- `<output_dir>/p4_mesh/object_mesh.ply` (Stage 4 出力)
- `<output_dir>/p2_colmap/camera_poses.json` (Stage 2 出力)
- `<output_dir>/p2_colmap/intrinsics.json` (Stage 2 出力, 任意 — 存在時はグリッドサーチをスキップ。`dist_coeffs` を含む場合はフレームをアンディストーションする)
- `<output_dir>/p1_frames/*.jpg`
- `<output_dir>/p3_masks/masks/*.png`
- `<output_dir>/p2_colmap/colmap_sparse/0/cameras.bin` (任意 — `intrinsics.json` に歪み情報が無い場合のフォールバック)

主出力:

- `<output_dir>/p2_colmap/intrinsics.json` (推定モード時のみ書き戻し)
- `<output_dir>/p5_texture/texture.png`
- `<output_dir>/p5_texture/textured_mesh.mtl`
- `<output_dir>/p5_texture/textured_mesh.obj`

## 詳細フロー

1. メッシュと pose 読み込み、対応フレーム index を解決。
2. 内部パラメータ取得 (3 段階の解決順):
   1. `intrinsics.json` が存在しスキーマ・解像度が一致すれば直接ロード (`_maybe_load_intrinsics`)。`source` が `colmap:*` で `dist_coeffs` が欠落している旧 JSON は、`cameras.bin` から自動で歪み係数を補填して再保存する (旧バージョンとの互換)
   2. 不在なら **COLMAP `cameras.bin` から直接復元** (`_maybe_load_colmap_intrinsics`) → 復元値を `intrinsics.json` に書き戻す
   3. それも失敗したら FOV **25°-90°** のグリッドサーチ → Nelder-Mead で `(fx, fy, cx, cy)` を最適化、`source: "estimated"` タグ付きで保存 (歪みは推定しない)
   - 解像度ベースの自動 cap は `TEXTURE_MAX_SIZE` (デフォルト 2048) で auto モード時のみ適用
3. **フレームアンディストーションのセットアップ**: `intrinsics.dist_coeffs` に非ゼロ係数が含まれる場合、`cv2.initUndistortRectifyMap` で `K` と OpenCV 形式 `[k1, k2, p1, p2, k3]` から remap マップを 1 度だけ生成し、`_FrameCache` に保持する。以降フレームは `cv2.remap(LINEAR)`、SAM2 マスクは `cv2.remap(NEAREST)` でアンディストーションされて返る。これにより以降の段階は純粋ピンホール投影のままで OK。
4. `xatlas.parametrize` で UV 展開 (並列化: `scripts/texture/parallel_uv.py` により空間分割で最大8ワーカー並列実行)。
   - 入力メッシュが `TEXTURE_UV_MAX_FACES` (デフォルト 300,000) を超える場合は Open3D `simplify_quadric_decimation` で UV proxy を生成。Stage 4 のメッシュ簡略化が無効化されたケースの保険
5. 全テクセルで Top-K ビューをスコアリング。
   - スコア: 法線角度 + 距離 + 可視性（簡易Zテスト）+ SAM2マスク
   - `top-1` と `top-2` が拮抗し、かつ視点差が大きい texel を conflict 候補として検出
   - `TEXTURE_VIEW_ASSIGN_MODE=legacy` では conflict が多い face を face 単位で dominant view に固定
   - `TEXTURE_VIEW_ASSIGN_MODE=region_gc` では conflict face を連続曲面 region にまとめ、region 内の face label を最適化
6. Top-K 候補ビューの色整合性を評価する。
   - 候補色の分散が大きい texel は、透明物体・光沢物体・反射でビュー間の見えが一致していないとみなし、Top-1 ビューへ harden する。
7. non-conflict texel は Top-K ビューをスコア加重ブレンド。
   - conflict face / region は single-view、その他は上位 K 個のビューから色を決定
   - 未充填テクセルは relaxed 閾値で全ビュー再スキャンしフォールバック
8. `region_gc` では narrow seam leveling で view 境界を局所的に平滑化。
   - `TEXTURE_QUALITY_BOOST` 有効時は boundary component ごとに複数補助ビューを比較し、
     色正規化 + ECC/phase correlation 整列つきで最良候補の detail を注入する。
9. UV seam 周辺を反復補間して隙間埋め。
10. 必要なら supersample -> downsample、sharpen を適用して PNG 書き出し。
11. OBJ/MTL と診断情報 `p5_texture/diagnostics.json` を生成。

## アルゴリズム要点

- 視点選定:
  - `normal・view_dir` の余弦に指数 (`TEXTURE_ANGLE_EXP`) を適用
  - 距離減衰 (`TEXTURE_DIST_POW`) を適用
  - 可視性はビューごとの深度ラスタライズで判定
- 貼り付け:
  - 競合が弱い領域はテクセル単位 Top-K ブレンド
  - Top-K 候補色が大きく食い違う texel は透明・光沢・反射による view-dependent な見えと判断し、Top-1 single-view に寄せる
  - `legacy`: 円筒面のような競合が強い領域は face 単位の single-view 貼り付けに切り替え
  - `region_gc`: 円筒面のような競合が強い領域は face graph を region 化し、region 内の label を揃えて single-view 貼り付けに切り替え
  - `region_gc` の境界は narrow seam leveling を挟み、hard seam を少し抑えてから seam padding に入る
  - `TEXTURE_QUALITY_BOOST` は Top-K 候補から複数の補助ビューを評価し、component 単位で最も整合する detail source を選ぶ
  - 補助ビューは局所色正規化後に phase correlation と ECC で平行移動整列し、低周波と高周波を分けて merge する
  - 未充填テクセルは relaxed 閾値でフォールバックしてから seam padding
- マスク考慮:
  - 投影点が SAM2 マスク内にあるサンプルのみ採用
  - SAM2 マスクもフレームと同じ remap でアンディストーションされる (`cv2.INTER_NEAREST`)。bilinear だとシルエット沿いに半端な縁テクセルが生まれ、背景画素がマスクテストを通ってしまうため
- カメラモデル整合:
  - COLMAP の bundle adjustment は SIMPLE_RADIAL 等の歪みモデル前提で世界座標を解いている。Stage 5 内部の投影は純粋ピンホールで動くため、フレームを `cv2.remap` で先にアンディストーションしてから常用する経路に揃えてある
  - 歪みを無視してピンホール投影を歪み付き画像に当てると、円筒・カップなど曲面上で隣接テクセルがそれぞれ違うサブピクセル量だけ位置ずれし、Top-K ブレンドで詳細が打ち消されて斑点状に見える

## パラメータ

| 名前 | 既定値 | 説明 |
|---|---:|---|
| `TEXTURE_SIZE` | `0` | 最終テクスチャ解像度。`0` 以下は `round(sqrt(video_width * video_height))` の正方形を自動適用 |
| `TEXTURE_MAX_SIZE` | `2048` | auto モード時の上限。`0` で無制限。`TEXTURE_SIZE>0` (manual) はバイパス |
| `TEXTURE_UV_MAX_FACES` | `300000` | xatlas 入力の上限 (Stage 4 簡略化が無効化されたケースの保険)。`0` で無制限 |
| `TEXTURE_VIEW_ASSIGN_MODE` | `region_gc` | view 割当モード。`legacy` は従来の face lock、`region_gc` は曖昧な連続曲面を region 最適化して single-view 化 |
| `TEXTURE_OVERSAMPLE` | `2` | 内部解像度倍率 |
| `TEXTURE_MIN_COS` | `0.2` | 面法線と視線方向の最小余弦 |
| `TEXTURE_ANGLE_EXP` | `4.0` | 角度重み指数 |
| `TEXTURE_DIST_POW` | `1.0` | 距離減衰指数 |
| `TEXTURE_SHARPEN` | `0.15` | 最終アンシャープ量 |
| `TEXTURE_BLEND_TOPK` | `3` | テクセルあたりブレンドするビュー数 (1=ブレンドなし) |
| `TEXTURE_BLEND_HARD_RATIO` | `2.0` | top-1 / top-2 スコア比がこの値を超えるテクセルはシングルビュー化 (0=無効) |
| `TEXTURE_COLOR_HARDENING` | `true` | Top-K 候補色の不一致が大きいテクセルを single-view 化する |
| `TEXTURE_COLOR_HARDENING_THRESHOLD` | `0.18` | single-view 化する RGB spread しきい値 ([0,1] 正規化 RGB の加重 RMS 距離) |
| `TEXTURE_QUALITY_BOOST` | `false` | `region_gc` 向けの高品質境界 refinement。複数補助ビュー比較、局所色正規化、ECC 整列、detail 注入を有効化 |

内部固定しきい値:

- `conflict_ratio = 1.35`
- `conflict_view_angle_deg = 20`
- `conflict_face_min_texels = 4`
- `conflict_face_min_frac = 0.2`
- `conflict_face_min_coverage = 0.7`
- `conflict_smooth_dot = 0.95`
- `conflict_smooth_gain = 1.05`
- `conflict_smooth_min_neighbors = 2`

既定値の定義元:

- 公開パラメータ: `scripts/config_defaults.py`
- conflict lock の内部しきい値: `scripts/stage_texture_bake.py`

## パフォーマンス最適化

テクスチャベイク処理には以下の最適化が適用されている。

- **A. メモリ適応型 LRU キャッシュ**: フレーム画像とマスクを `_FrameCache` でキャッシュ。利用可能メモリからキャッシュ容量を自動算出し、フレーム 70% / マスク 30% の予算比率で確保する。
- **B. GPU 深度ラスタライズ** (`scripts/texture/gpu_raster.py`): nvdiffrast による GPU アクセラレーテッドラスタライゼーションで深度バッファを生成。CPU フォールバックあり。
- **C. 深度バッファ LRU キャッシュ**: 視点ごとの深度バッファを `@lru_cache` でキャッシュし、同一視点の再計算を回避する。
- **D. 並列 UV アトラス生成** (`scripts/texture/parallel_uv.py`): メッシュを空間分割し、最大8ワーカーで並列に xatlas UV 展開を実行。`_TEXTURE_UV_PARALLEL_MIN_TOTAL_FACES` (10,000) 以下のメッシュでは並列化をスキップ。
- **E. 内在パラメータ推定の並列化**: `ThreadPoolExecutor` で複数フレームの色スコア評価を並列実行する。
- **F. COLMAP 内部パラメータ直接利用**: `intrinsics.json` (COLMAP 出力) が存在する場合、FOV グリッドサーチ + Nelder-Mead 最適化をスキップし、推定時間を大幅に短縮する。`intrinsics.json` が消失していても `colmap_sparse/0/cameras.bin` を直接読みに行く防御 fallback あり (`_maybe_load_colmap_intrinsics`)。COLMAP のバンドル調整値はテクスチャベイクの正確さに直結するため、特に円筒形・自己類似性のあるオブジェクト (例: カップヌードル) では推定値で代替できない。
- **G. テクセル UV 座標の事前計算**: 三角形ごとの UV 座標を `np.stack` で事前計算し、`fillConvexPoly` で再利用する。
- **H. OBJ バッファ書き出し**: OBJ ファイルを文字列リストに蓄積してから一括書き込みする。

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

## 既知の落とし穴: 歪み係数を捨てたピンホール投影による曲面ディテール消失

`intrinsics.json` に `dist_coeffs` が無い (旧バージョン由来) か、すべて 0 のとき、Stage 5 はフレームをアンディストーションせず純粋ピンホール投影でサンプルする。COLMAP がモデル `SIMPLE_RADIAL` で解いている場合は、世界座標自体が k1 込みで bundle adjustment された値であるため、ピンホール投影と歪み付き画像の組み合わせは整合しない。

サブピクセル位置ずれは画像中心からの距離 r に応じて増え、k1≈0.03 の iPhone キャプチャでは中心から離れた領域で 2px 以上に達する。フラット箱型 (Cereal 等) では同一面のテクセルが揃って同方向にシフトするため見た目には影響しないが、円筒・カップ型 (Coffee_Can / Jagarico 等) ではテクセルごとにシフト量が変わるため、Top-K ブレンドで細部が打ち消し合って色斑点として現れる。

検出と回避:

- `intrinsics.json` に `dist_coeffs` フィールドがあり全 0 でないことを確認する。
- 旧 JSON で欠落していても、`source` が `colmap:*` であれば Stage 5 が起動時に `cameras.bin` から自動補填する (`scripts/texture/bake.py`)。
- `colmap:SIMPLE_RADIAL` 由来であれば `[k1, 0, 0, 0, 0]` の 5 要素配列が、`OPENCV` 由来であれば `[k1, k2, p1, p2, k3]` が入る。
- `_FrameCache.undistort_enabled = True` のとき Stage 5 ログに `Frame undistortion: ON (model=..., dist=[...])` が出る。

なお `_estimate_intrinsics` 経由の値 (`source: "estimated"`) は歪みを推定しないため `dist_coeffs` を持たない。歪みのある実機キャプチャに対してはあくまで COLMAP 経路を使うのが前提。

## 既知の落とし穴: 推定 intrinsics による細部消失

円筒形 (カップ缶など) や自己類似性のあるテクスチャ (繰り返しパターン) では、`_estimate_intrinsics()` の Nelder-Mead 最適化が **sub-pixel スケールで歪んだ局所最適** に収束することがある。フラットな箱型のオブジェクトでは同程度の誤差は問題ないが、Top-K=3 ビューブレンドと組み合わさると細かいテキスト・絵柄が完全消失する。

判別方法: `intrinsics.json` の `source` フィールドを確認する。

| `source` 値 | 経路 | テクスチャ品質への影響 |
|---|---|---|
| `colmap:SIMPLE_RADIAL` (など) | Stage 2 が直接書いた COLMAP の bundle adjustment 値 | ✓ 推奨 |
| `estimated` | テクスチャ stage の grid search + Nelder-Mead | ⚠ 要注意 (細部消失リスク) |
| (無し) | 古い `_estimate_intrinsics` 出力 (タグなし) | ⚠ 要注意 |

**回避策**:

1. Stage 2 (COLMAP SfM) を完了させ、`<output_dir>/intrinsics.json` が `source: "colmap:..."` 付きで生成されることを確認する。
2. Stage 5 リスタート時に `intrinsics.json` が削除される問題は `scripts/dashboard/checkpoints.py` で対処済 (cleanup_files から除外)。
3. `intrinsics.json` 不在でも `colmap_sparse/0/cameras.bin` があれば `_maybe_load_colmap_intrinsics` が自動復元する。
4. それでも grid search に落ちる場合、FOV 範囲は 25°-90° に拡張済み (狭い FOV close-up shot にも対応)。

## 参考文献

- xatlas repository: <https://github.com/jpcy/xatlas>
- nvdiffrast: <https://github.com/NVlabs/nvdiffrast>
- Nelder-Mead (SciPy optimize.minimize): <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html>
