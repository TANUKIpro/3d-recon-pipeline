# colmap_sfm

## 対象タスク

- Stage 2: COLMAP Structure-from-Motion
- 実装: `scripts/stage_colmap_sfm.py`
- 呼び出し元: `scripts/dashboard/pipeline_runner.py`, `scripts/pipeline.py`

## 概要

抽出されたフレーム画像から COLMAP を用いて特徴抽出・マッチング・スパース再構成を行い、カメラポーズ (外部パラメータ) とカメラ内部パラメータを推定する。

## 入出力関係

前段入力:

- `<output_dir>/frames/*.jpg` (Stage 1 出力)

主出力:

- `<output_dir>/camera_poses.json` — カメラ外部パラメータ (c2w 4x4 行列)
- `<output_dir>/intrinsics.json` — カメラ内部パラメータ (fx, fy, cx, cy, K 行列)
- `<output_dir>/colmap_sparse/` — COLMAP スパース再構成 (バイナリ形式)
- `<output_dir>/colmap_sparse_points.ply` — スパース点群

後段利用:

- Stage 4 (`gs2mesh_reconstruct`) が `colmap_sparse/` を 3DGS 学習と画像アンディストーションに使用
- Stage 5 (`texture_bake`) が `camera_poses.json` + `intrinsics.json` を使用
- Stage 6 (`post_texture_contact_cleanup`) が `camera_poses.json` + `intrinsics.json` を使用

## 詳細フロー

1. **特徴抽出** (`colmap feature_extractor`):
   - SIFT 特徴点を検出。`--SiftExtraction.max_num_features` で上限制御。
   - `COLMAP_DSP_SIFT=true` の場合、`estimate_affine_shape` + `domain_size_pooling` を有効化 (DSP-SIFT)。
   - `--ImageReader.single_camera 1` により全フレームで共有カメラモデルを使用。

2. **特徴マッチング** (`colmap exhaustive_matcher` / `sequential_matcher`):
   - `COLMAP_MATCHER=exhaustive`: 全ペアマッチング (精度重視)。
   - `COLMAP_MATCHER=sequential`: 時間的に近いフレーム間のみマッチング (高速)。

3. **スパース再構成** (`colmap mapper`):
   - バンドル調整付きインクリメンタル SfM でカメラポーズを推定。
   - 再構成結果は `colmap_sparse/0/` に出力 (images.bin, cameras.bin, points3D.bin)。

4. **カメラポーズ書き出し**:
   - `images.bin` から quaternion + translation を読み込み。
   - world-to-camera → camera-to-world (c2w) 4x4 行列に変換。
   - フレーム名と対応付けて `camera_poses.json` に保存。

5. **内部パラメータ書き出し**:
   - `cameras.bin` から最も多くの画像に使われているカメラモデルを選択。
   - 10種類のカメラモデル (SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV 等) から fx/fy/cx/cy を抽出。
   - `intrinsics.json` に保存。テクスチャベイキングの内部パラメータ推定をスキップ可能にする。

6. **スパース点群書き出し** (`colmap model_converter --output_type PLY`):
   - 再構成されたスパース点群を PLY 形式で保存。

## パラメータ

| 名前 | 既定値 | 説明 |
|---|---:|---|
| `COLMAP_MATCHER` | `exhaustive` | マッチング方式 (`exhaustive` / `sequential`) |
| `COLMAP_MAX_FEATURES` | `32768` | SIFT 特徴点の最大数 |
| `COLMAP_IMAGE_SIZE` | `2048` | 特徴抽出時の最大画像サイズ (px) |
| `COLMAP_USE_GPU` | `false` | 特徴抽出・マッチングでの GPU 使用 |
| `COLMAP_DSP_SIFT` | `true` | DSP-SIFT (アフィン形状推定 + ドメインサイズプーリング) |
| `COLMAP_FIRST_OCTAVE` | `-1` | SIFT の最初のオクターブ (-1 = upsampled) |

既定値の定義元: `scripts/config_defaults.py`

## 失敗時の典型原因

- `COLMAP mapper produced no valid reconstruction`: 十分な特徴マッチングが得られなかった (フレーム数不足、テクスチャの乏しいシーン等)
- `Unsupported COLMAP camera model`: COLMAP が稀なカメラモデルを選択した場合

## 参考文献

- COLMAP: <https://colmap.github.io/>
- COLMAP GitHub: <https://github.com/colmap/colmap>
- DSP-SIFT: Dong & Soatto, "Domain-Size Pooling in Local Descriptors", CVPR 2015
