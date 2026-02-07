# im2pc-pipeline

RGB動画から テクスチャ付き3Dメッシュ (OBJ) を生成する Docker 完結型パイプライン。

SAM2 によるインタラクティブ物体セグメンテーション、Pi3X による多視点3D再構成、DiffCD による暗黙表面メッシュ化、マルチビューテクスチャベイキングを1コンテナで実行する。

## パイプライン概要

```
入力: RGB動画 (.mp4)
  │
  ├─ Stage 1: フレーム抽出 (CPU)
  │    動画から等間隔にフレームを JPEG 抽出
  │
  ├─ Stage 2: SAM2 セグメンテーション (GPU)
  │    Gradio UI で対象物体をクリック → 全フレームにマスク伝播
  │
  ├─ Stage 3: Pi3X 3D再構成 (GPU)
  │    全フレーム一括推論 → トリプルフィルタ (信頼度 + 深度エッジ + SAM2マスク)
  │
  ├─ Stage 4: 点群デノイズ (CPU)
  │    DBSCAN クラスタリング + Statistical Outlier Removal
  │
  ├─ Stage 5: DiffCD メッシュ再構成 (GPU/JAX)
  │    暗黙表面フィッティング → Marching Cubes → Laplacian 平滑化
  │
  └─ Stage 6: テクスチャベイキング (CPU)
       カメラ内部パラメータ推定 → xatlas UV展開 → マルチビューテクスチャ投影

出力: textured_mesh.obj / .mtl / texture.png
```

## 動作環境

| 項目 | 要件 |
|------|------|
| GPU | NVIDIA (CUDA Compute ≥ 7.0), VRAM 16GB 推奨 |
| Docker | 20.10 以上 + Docker Compose v2 |
| NVIDIA Container Toolkit | nvidia-docker2 または nvidia-container-toolkit |
| OS | Linux (Ubuntu 22.04 で検証済み) |

## クイックスタート

```bash
# 1. リポジトリクローン
git clone <repo-url> && cd im2pc-pipeline

# 2. Docker イメージビルド (初回 15〜30分)
docker compose build

# 3. 入力動画を配置
cp /path/to/video.mp4 data/input/

# 4. パイプライン実行
./run.sh data/input/video.mp4
```

Stage 2 で Gradio UI が起動するので、ブラウザで http://localhost:7860 を開き、対象物体をクリックして「Confirm & Propagate」を押す。残りのステージは自動で進行する。

## 使い方

### 基本実行

```bash
# run.sh 経由 (推奨)
./run.sh data/input/video.mp4

# docker compose 直接
docker compose run --rm --service-ports \
  pipeline /app/scripts/pipeline.py /data/input/video.mp4
```

### パラメータ変更

環境変数で各ステージの挙動を制御できる:

```bash
docker compose run --rm --service-ports \
  -e MAX_FRAMES=20 \
  -e PIXEL_LIMIT=150000 \
  -e SAM2_MODEL=small \
  -e DIFFCD_RESOLUTION=384 \
  pipeline /app/scripts/pipeline.py /data/input/video.mp4
```

### 途中再開

中断後にステージ N から再開:

```bash
docker compose run --rm --service-ports \
  pipeline /app/scripts/pipeline.py /data/input/video.mp4 --skip-to 4
```

### 個別ステージ実行

```bash
# フレーム抽出のみ
docker compose run --rm pipeline \
  /app/scripts/stage_extract_frames.py /data/input/video.mp4 /data/output

# デノイズのみ
docker compose run --rm pipeline \
  /app/scripts/stage_denoise.py /data/output/object.ply /data/output
```

## 環境変数

### フレーム抽出 (Stage 1)

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `FRAME_INTERVAL` | `10` | N フレームごとに1枚抽出 |
| `MAX_FRAMES` | `50` | 抽出フレーム数の上限 |

### SAM2 セグメンテーション (Stage 2)

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `SAM2_MODEL` | `large` | モデルサイズ: `tiny` / `small` / `base` / `large` |

### Pi3X 3D再構成 (Stage 3)

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `PIXEL_LIMIT` | `255000` | フレームあたり最大ピクセル数 (リサイズ閾値) |
| `CONFIDENCE_THRESHOLD` | `0.1` | 信頼度フィルタ閾値 |
| `EDGE_RTOL` | `0.03` | 深度エッジフィルタの相対許容値 |

### DiffCD メッシュ再構成 (Stage 5)

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `DIFFCD_BATCH_SIZE` | `3000` | バッチサイズ |
| `DIFFCD_N_BATCHES` | `25000` | 学習バッチ総数 |
| `DIFFCD_RESOLUTION` | `384` | Marching Cubes 解像度 |

### テクスチャベイキング (Stage 6)

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `TEXTURE_SIZE` | `2048` | テクスチャアトラスの解像度 (px) |

## 出力ファイル

パイプライン完了後、`data/output/` に以下が生成される:

```
data/output/
├── textured_mesh.obj      # 最終成果物: テクスチャ付き3Dメッシュ
├── textured_mesh.mtl      # マテリアル定義
├── texture.png            # テクスチャアトラス (2048x2048)
├── object_mesh.ply        # DiffCD出力メッシュ (平滑化済み)
├── object_denoised.ply    # デノイズ済み点群
├── object.ply             # Pi3Xトリプルフィルタ済み点群
├── camera_poses.json      # カメラ外部パラメータ (4x4行列)
├── intrinsics.json        # 推定カメラ内部パラメータ
├── frames/                # 抽出フレーム画像 (JPEG)
├── masks/                 # SAM2セグメンテーションマスク (PNG)
└── diffcd/                # DiffCD作業ディレクトリ
```

## アーキテクチャ

### トリプルフィルタ (Stage 3)

Pi3X は全画像に対して推論し、3段階のフィルタで対象物体の点群を抽出する:

1. **信頼度フィルタ** — Pi3X の出力信頼度が閾値未満の点を除去
2. **深度エッジフィルタ** — 深度の不連続箇所 (物体境界のアーティファクト) を除去
3. **SAM2 マスクフィルタ** — セグメンテーションマスク外の点を除去

### PyTorch / JAX 共存

DiffCD (JAX) と SAM2/Pi3X (PyTorch) は GPU コンテキストが競合するため、DiffCD はサブプロセスとして実行される。`XLA_PYTHON_CLIENT_PREALLOCATE=false` により JAX の事前メモリ確保を無効化し、PyTorch との共存を実現している。

### VRAM 管理

RTX 4090 (16GB) で全ステージを動作させるため、各 GPU ステージ終了時にモデルを明示的に解放し `torch.cuda.empty_cache()` でメモリを回収する。

## VRAM とパフォーマンス

RTX 4090 Laptop (16GB) での実測値:

| ステージ | VRAM ピーク | 所要時間 |
|----------|-----------|---------|
| SAM2 (large) | ~2 GB | ~15秒 (伝播) |
| Pi3X (20フレーム, 150Kpx) | ~14 GB | ~1分 |
| DiffCD (res=384, 25Kバッチ) | ~10 GB | ~10分 |
| デノイズ / テクスチャ | CPU のみ | ~1分 |

> **注意**: デフォルト設定 (`MAX_FRAMES=50`, `PIXEL_LIMIT=255000`) では RTX 4090 16GB で OOM が発生する場合がある。16GB GPU では `MAX_FRAMES=20 PIXEL_LIMIT=150000` を推奨。

## Docker イメージ構成

```
nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
├── Python 3.11 + system deps (ffmpeg, OpenGL)
├── PyTorch 2.5+ (CUDA 12.1 wheels)
├── Python deps (numpy<2.0, opencv, gradio, scipy, trimesh, xatlas, ...)
├── wandb + scikit-image (DiffCD 依存)
├── SAM2 (editable install, SAM2_BUILD_CUDA=0)
├── Pi3X (PYTHONPATH 経由)
├── DiffCD (パッチ適用済み: tyro float fix + jax.tree.map)
├── JAX CUDA プラグイン修正 (CuDNN ≥9.8.0)
└── Pipeline スクリプト (/app/scripts/)
```

イメージサイズ: 約 21 GB

## トラブルシューティング

### Pi3X で OOM が発生する

`MAX_FRAMES` と `PIXEL_LIMIT` を下げる:

```bash
docker compose run --rm --service-ports \
  -e MAX_FRAMES=15 -e PIXEL_LIMIT=120000 \
  pipeline /app/scripts/pipeline.py /data/input/video.mp4
```

### SAM2 の Gradio UI に接続できない

`--service-ports` フラグを忘れていないか確認。`run.sh` には含まれている。

```bash
docker compose run --rm --service-ports pipeline ...
```

### DiffCD が遅い / 解像度を上げたい

`DIFFCD_RESOLUTION=512` は A100 40GB 以上が必要。16GB GPU では `384` が上限。

### ビルドが失敗する

NVIDIA Container Toolkit が正しくインストールされているか確認:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 nvidia-smi
```

## ライセンス

本リポジトリのスクリプトは MIT ライセンス。依存プロジェクトは各自のライセンスに従う:

- [SAM2](https://github.com/facebookresearch/sam2) — Apache 2.0
- [Pi3X](https://github.com/yyfz/Pi3) — MIT
- [DiffCD](https://github.com/Linusnie/diffcd) — MIT
