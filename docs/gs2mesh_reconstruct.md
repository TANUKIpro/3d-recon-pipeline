# gs2mesh_reconstruct

## 対象タスク

- Stage 4: gs2mesh 再構成
- 実装:
  - `scripts/stage_gs2mesh_reconstruct.py` (パイプラインオーケストレーション)
  - `scripts/gpu_tsdf.py` (GPU TSDF Fusion)
  - `scripts/gs2mesh_config.py` (設定・プリセット管理)
  - `scripts/vram_tier.py` (VRAM ティア自動検出)
- 呼び出し元: `scripts/dashboard/pipeline_runner.py`, `scripts/pipeline.py`

## 概要

3D Gaussian Splatting で学習したシーン表現からステレオ深度を推定し、GPU TSDF Fusion でボリューム統合してメッシュを抽出する。COLMAP 出力と SAM2 マスクを入力とし、単一のメッシュ `object_mesh.ply` を出力する。

## 入出力関係

前段入力:

- `<output_dir>/p1_frames/*.jpg` (Stage 1 出力)
- `<output_dir>/p2_colmap/colmap_sparse/` (Stage 2 出力)
- `<output_dir>/p3_masks/masks/*.png` (Stage 3 出力, optional)

主出力:

- `<output_dir>/p4_mesh/object_mesh.ply` — 再構成メッシュ

中間ファイル:

- `<output_dir>/p4_mesh/gs2mesh_workspace/` — 3DGS チェックポイント、アンディストーション結果
- `p4_mesh/gs2mesh_workspace/splatting_output/` — 3DGS モデル (point_cloud.ply)
- `p4_mesh/gs2mesh_workspace/debug/` — ステレオ推定のリトライログ・メタデータ

後段利用:

- Stage 5 (`texture_bake`) が `p4_mesh/object_mesh.ply` を入力
- Stage 6 (`post_texture_contact_cleanup`) が `p4_mesh/object_mesh.ply` を接地平面推定に使用

## 詳細フロー

### Step 0: 前処理

1. **画像アンディストーション** (`colmap image_undistorter`):
   - COLMAP の歪みモデル (SIMPLE_RADIAL 等) を PINHOLE に変換。3DGS が要求する無歪み画像を生成。

2. **COLMAP モデル変換** (`colmap model_converter`):
   - バイナリ形式 → テキスト形式。gs2mesh の Renderer が `images.txt` / `cameras.txt` を読み込むため。

3. **gs2mesh データディレクトリ構築**:
   - アンディストーション出力へのシンボリックリンクを作成 (`images/`, `sparse/`)。

### Step 1: 3D Gaussian Splatting 学習

- gaussian-splatting の `train.py` をサブプロセスとして実行。
- **SparseGaussianAdam** オプティマイザを使用 (利用可能な場合、`auto` プロファイル)。
- `compat` プロファイルではデフォルトオプティマイザにフォールバック。
- 既存チェックポイントがある場合は再利用 (再学習をスキップ)。

### Step 2: gs2mesh ステレオ深度推定

- gs2mesh の `run_single.py` をサブプロセスとして実行。
- 学習済み 3DGS モデルからステレオ画像ペアをレンダリング。
- **DLNR (Middlebury)** ステレオマッチングモデルで視差 → 深度マップを推定。
- `--skip_TSDF` フラグにより gs2mesh 内蔵の CPU TSDF は使用せず、深度推定のみ実行。

**リトライ戦略**: `auto` プロファイルで失敗した場合、`compat` プロファイル (CUDA_LAUNCH_BLOCKING=1 + compat rasterizer) で自動リトライ。

**分類されるエラー**: `cuda_illegal_memory_access`, `cuda_rasterizer_failure`, `renderer_signature_mismatch`, `missing_rasterizer_extension`, `missing_output_artifacts`

### Step 3: SAM2 マスク統合

- canonical SAM2 マスク (`masks/*.png`) を gs2mesh のビューディレクトリに `left_mask.npy` として配置。
- カメラデータ JSON からフレームインデックスを解決し、対応するマスクをリサイズ・変換。
- `GS2MESH_USE_MASKS=true` かつ `mask_dir` が渡された場合、Stage 4 は TSDF 実行前に必ずこの変換を行う。

### Step 4: GPU TSDF Fusion

- **Open3D `VoxelBlockGrid`** (CUDA または CPU) でボリューム統合。
- gs2mesh 標準の CPU `ScalableTSDFVolume` を置き換え、大幅な高速化を実現。
- 各フレームの深度マップ + RGB + マスクを統合:
  1. 深度マップ読み込み + オブジェクトマスク適用 (erosion + morphological closing)
  2. オクルージョンマスク適用
  3. 深度しきい値処理 (min/max baselines)
  4. SAM2 mask depth mode 適用 (`crop` / `fill` / `replace`)
  5. カメラ行列構築 (extrinsic, intrinsic)
  6. `VoxelBlockGrid.integrate()` で TSDF 更新
- 有効な深度ピクセルがないフレームはスキップ。

**SAM2 mask depth mode**:

- `crop` (default): SAM2 mask で stereo depth を切り抜く。visual hull は構築しない。
- `fill`: `left_mask.npy` 群から visual hull を構築し、SAM2 mask 内かつ stereo depth が無効または min/max 範囲外の画素だけ補完する。
- `replace`: visual hull depth を SAM2 mask 内の主ソースにする。transparent/specular surface の誤った stereo depth と occlusion mask は mask 内では信用しない。
- `replace` で visual hull を構築できない場合は旧TSDFへ黙って戻らず、明示的にエラーにする。

**OOM フォールバック梯子** (GPU TSDF):

| レベル | 変更内容 |
|--------|---------|
| requested | ユーザー指定パラメータ |
| reduced_vram | voxel_size ≥ 0.005, depth_trunc ≥ 0.04, baselines ≤ 20, dilate ≥ 2, block_count ≤ 100K |
| safe_vram | baselines ≤ 16, dilate ≥ 3, block_count ≤ 75K |

### Step 5: メッシュ抽出 + デシメーション

- Marching Cubes でメッシュ抽出 → `orient_triangles()` でワインディング修正 → 小クラスタ除去 (`tsdf_cleaning_threshold`)。
- **メッシュ簡略化** (`MESH_DECIMATION` が有効時): Open3D `simplify_quadric_decimation()` で目標三角形数までデシメートする。Garland-Heckbert QEM がベース。VoxelBlockGrid 由来の頂点色 (RGB) が保持されるため、Stage 5 のキャップ検出 (`cap_region.py`) も従来通り動作する。
- **シルエット IoU セーフネット**: 簡略化前後のメッシュを 8 視点 (`MESH_DECIMATION_IOU_VIEWS`) で `o3d.t.geometry.RaycastingScene` レンダリング → binary silhouette IoU を計算。閾値 (`MESH_DECIMATION_MIN_IOU`, デフォルト 0.985) を割れば原メッシュへ自動 rollback。缶のリングや皿の縁などの薄物が消えた場合の保険。
- `object_mesh.ply` としてコピー。

**ターゲット三角形数の解決順序** (`scripts/vram_tier.py::mesh_decimation_target`):

1. `MESH_DECIMATION=0` なら 0 (無効化)
2. `MESH_TARGET_FACES=<N>` env なら N
3. それ以外は VRAM ティア × プリセット表から決定:

| ティア | default preset | high preset |
|---|---:|---:|
| t18 / t16 / t12 (GPU TSDF) | 200,000 | 500,000 |
| t8 / t_other (CPU TSDF) | 80,000 | 200,000 |

ターゲットを上回るときのみデシメーションを実行 (`if n_tris <= target: skip`)。

## プリセット

| プリセット | GS iterations | TSDF depth_trunc | cleaning threshold | 用途 |
|-----------|---:|---:|---:|---|
| `default` | 5,000 | 0.04 | 100,000 | バランス重視 |
| `high` | 15,000 | 0.03 | 50,000 | 高品質 (時間増) |
| `custom` | ユーザー定義 | ユーザー定義 | ユーザー定義 | 詳細チューニング |

## VRAM ティアシステム

`scripts/vram_tier.py` が起動時に GPU VRAM を自動検出:

| ティア | VRAM (MB) | TSDF デバイス |
|--------|---:|---|
| t18 | ≥ 17,000 | CUDA:0 |
| t16 | ≥ 14,000 | CUDA:0 |
| t12 | ≥ 10,000 | CUDA:0 |
| t8 | ≥ 7,000 | CPU:0 (自動フォールバック) |
| t_other | < 7,000 | CPU:0 (自動フォールバック) |

`VRAM_TIER_OVERRIDE` 環境変数でティアを手動指定可能。

## ランタイムプロファイル

| プロファイル | オプティマイザ | 特徴 |
|-------------|-------------|------|
| `auto` | SparseGaussianAdam (利用可能時) | 高速、最新の gaussian-splatting ビルド向け |
| `compat` | デフォルトオプティマイザ | CUDA_LAUNCH_BLOCKING=1 + compat rasterizer overlay で安定性重視 |

## パラメータ

### 公開パラメータ (Dashboard / docker-compose.yml)

| 名前 | 既定値 | 説明 |
|---|---:|---|
| `GS2MESH_PRESET` | `default` | プリセット名 |
| `GS2MESH_GS_ITERATIONS` | `5000` | 3DGS 学習イテレーション数 |
| `GS2MESH_RUNTIME_PROFILE` | `auto` | ランタイムプロファイル |
| `GS2MESH_STEREO_MODEL` | `DLNR_Middlebury` | ステレオ深度推定モデル |
| `GS2MESH_TSDF_VOXEL_SIZE` | `0.005` | TSDF ボクセルサイズ (m) |
| `GS2MESH_TSDF_DEPTH_TRUNC` | `0.04` | TSDF 深度切断距離 (m) |
| `GS2MESH_USE_MASKS` | `true` | SAM2 マスクを TSDF 統合に使用 |
| `GS2MESH_MASK_DEPTH_MODE` | `crop` | `crop`: mask で stereo depth を切り抜き / `fill`: 無効 depth を visual hull で補完 / `replace`: SAM2 visual hull を mask 内 depth の主ソースにする |
| `MESH_DECIMATION` | `1` (有効) | `0` でメッシュ簡略化 + IoU セーフネットを完全に無効化 (旧挙動) |
| `MESH_TARGET_FACES` | (自動) | 簡略化後の目標三角形数。未指定なら VRAM ティア × プリセット表から解決 |
| `MESH_DECIMATION_MIN_IOU` | `0.985` | 簡略化前後のシルエット IoU 閾値。下回ると原メッシュへ自動 rollback |
| `MESH_DECIMATION_IOU_VIEWS` | `8` | IoU 検証に使う視点数。`0` で IoU セーフネットを無効化 (デシメーション結果を常に採用) |
| `GS2MESH_SILHOUETTE_VOXELS` | `160` (`high` は `224`) | visual hull carving の最大軸 voxel 数 |
| `GS2MESH_SILHOUETTE_MIN_VIEWS` | `3` | visual hull 内判定に必要な最小 view 数 |
| `GS2MESH_SILHOUETTE_CONSENSUS` | `0.85` | 投影された有効 view のうち mask 内である必要がある割合 |
| `GS2MESH_SILHOUETTE_MASK_DILATE_PX` | `2` | carving 用 mask の dilation 半径 |
| `GS2MESH_SAM2_PRIMARY_MAX_EXTRA_VIEWS` | `64` | SAM2 primary 時に追加する skipped-frame visual hull view の上限。`0` で追加無効 |

### 内部パラメータ

| 名前 | 既定値 | 説明 |
|---|---:|---|
| `GS2MESH_TSDF_SCALE` | `1.0` | 深度スケール乗数 |
| `GS2MESH_TSDF_MIN_DEPTH_BASELINES` | `4` | 最小深度ベースライン数 |
| `GS2MESH_TSDF_MAX_DEPTH_BASELINES` | `20` | 最大深度ベースライン数 |
| `GS2MESH_TSDF_DILATE` | `1` | フレーム間引き (1 = 全フレーム使用) |
| `GS2MESH_TSDF_CLEANING_THRESHOLD` | `100000` | 小クラスタ除去しきい値 (三角形数) |
| `GS2MESH_TSDF_USE_OCCLUSION_MASK` | `true` | オクルージョンマスクの使用 |
| `GS2MESH_TSDF_INVERT_MASK` | `false` | マスク反転 |
| `GS2MESH_TSDF_ERODE_MASK` | `true` | マスクの erosion 前処理 |
| `GS2MESH_TSDF_EROSION_KERNEL_SIZE` | `10` | erosion カーネルサイズ |
| `GS2MESH_TSDF_CLOSING_KERNEL_SIZE` | `10` | morphological closing カーネルサイズ |
| `GS2MESH_TSDF_BLOCK_COUNT` | `100000` | VoxelBlockGrid ブロック数 |
| `GS2MESH_TSDF_DEVICE` | `CUDA:0` | TSDF デバイス (VRAM ティアで上書き) |

既定値の定義元: `scripts/config_defaults.py`

## 失敗時の典型原因

- `cuda_illegal_memory_access`: GPU メモリ不正アクセス (VRAM 不足の兆候)
- `cuda_rasterizer_failure`: diff_gaussian_rasterization の CUDA エラー
- `renderer_signature_mismatch`: gaussian-splatting API 互換性問題 (Camera/rasterizer の引数不一致)
- `missing_rasterizer_extension`: diff_gaussian_rasterization モジュールが見つからない
- `missing_output_artifacts`: ステレオ推定の出力が不完全 (camera_data.json または depth.npy の欠落)
- GPU TSDF OOM: 全リトライ失敗後にエラー

## 参考文献

- gs2mesh: <https://github.com/yanivw12/gs2mesh> (Apache 2.0)
- gs2mesh 論文: "GS2Mesh: Surface Reconstruction from Gaussian Splatting via Novel Stereo Views" (ECCV 2024, [arXiv:2404.01810](https://arxiv.org/abs/2404.01810))
- 3D Gaussian Splatting: <https://github.com/graphdeco-inria/gaussian-splatting>
- 3DGS 論文: "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)
- DLNR: <https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network> (CVPR 2023)
- Open3D VoxelBlockGrid: <http://www.open3d.org/docs/release/python_api/open3d.t.geometry.VoxelBlockGrid.html>
