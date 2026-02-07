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
  │    Web ダッシュボードで対象物体をクリック → 全フレームにマスク伝播
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

# 4. Web ダッシュボード起動
docker compose up
```

ブラウザで **http://localhost:7860** を開くと Web ダッシュボードが表示される。

1. **Video** ドロップダウンで動画を選択
2. パラメータを必要に応じて調整 (Advanced Settings で詳細設定)
3. **Start Pipeline** をクリック
4. Stage 2 で SAM2 Canvas がアクティブになるので、対象物体を左クリック (除外は右クリック)
5. **Confirm & Propagate** で全フレームにマスク伝播 → 残りのステージは自動進行
6. ログ・進捗・3Dプレビューをリアルタイムで確認

## 使い方

### Web ダッシュボード (推奨)

```bash
# ダッシュボード起動
docker compose up

# バックグラウンド起動
docker compose up -d

# ログ確認 (バックグラウンド時)
docker compose logs -f

# 停止
docker compose down
```

ブラウザで http://localhost:7860 を開き、GUI から全操作を行う。

**ダッシュボード機能:**

- **動画選択**: `/data/input/` に配置した動画ファイルを自動検出
- **パラメータ設定**: 全ステージのパラメータを GUI から変更可能
- **SAM2 Canvas**: 左クリック = ポジティブポイント、右クリック = ネガティブポイント。Undo / Clear / Confirm & Propagate
- **進捗バー**: 6ステージの状態をリアルタイム表示 (pending → running → complete)
- **ログビューア**: WebSocket 経由でリアルタイムストリーミング
- **3D プレビュー**: three.js による点群・メッシュのインタラクティブ表示 (回転・ズーム)
- **フレームギャラリー**: 抽出フレーム・マスクのサムネイル一覧

### CLI 実行 (後方互換)

従来の CLI パイプラインも引き続き利用可能:

```bash
# ENTRYPOINT を上書きして CLI モードで実行
docker compose run --rm --service-ports \
  --entrypoint python3 \
  pipeline /app/scripts/pipeline.py /data/input/video.mp4

# パラメータ指定
docker compose run --rm --service-ports \
  --entrypoint python3 \
  -e MAX_FRAMES=20 \
  -e PIXEL_LIMIT=150000 \
  pipeline /app/scripts/pipeline.py /data/input/video.mp4

# 途中再開 (ステージ N から)
docker compose run --rm --service-ports \
  --entrypoint python3 \
  pipeline /app/scripts/pipeline.py /data/input/video.mp4 --skip-to 4
```

> **注意**: CLI モードでは Stage 2 で Gradio UI が起動する (従来と同じ動作)。

### 個別ステージ実行

```bash
# フレーム抽出のみ
docker compose run --rm --entrypoint python3 pipeline \
  /app/scripts/stage_extract_frames.py /data/input/video.mp4 /data/output

# デノイズのみ
docker compose run --rm --entrypoint python3 pipeline \
  /app/scripts/stage_denoise.py /data/output/object.ply /data/output
```

## 環境変数

### フレーム抽出 (Stage 1)

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `FRAME_INTERVAL` | `10` | N フレームごとに1枚抽出 |
| `MAX_FRAMES` | `50` | 抽出フレーム数の上限 (Pi3X入力でも上限として使用) |

### SAM2 セグメンテーション (Stage 2)

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `SAM2_MODEL` | `large` | モデルサイズ: `tiny` / `small` / `base` / `large` |

### Pi3X 3D再構成 (Stage 3)

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `PIXEL_LIMIT` | `255000` | フレームあたり最大ピクセル数 (必要時のみリサイズ) |
| `CONFIDENCE_THRESHOLD` | `0.1` | 信頼度フィルタ閾値 |
| `EDGE_RTOL` | `0.03` | 深度エッジフィルタの相対許容値 |
| `ALIGN_CAMERA_PLANE` | `1` | `1` でカメラ軌道平面を基準面(XZ)へ自動整列 (`0` で無効) |

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
├── camera_poses.json      # カメラ外部パラメータ + 使用フレームindex + 整列メタ情報
├── intrinsics.json        # 推定カメラ内部パラメータ
├── frames/                # 抽出フレーム画像 (JPEG)
├── masks/                 # SAM2セグメンテーションマスク (PNG)
└── diffcd/                # DiffCD作業ディレクトリ
```

## アーキテクチャ

### Web ダッシュボード

```
┌─ Browser (localhost:7860) ─────────────────────────────┐
│  Vanilla HTML/CSS/JS (ビルドツール不要)                   │
│  WebSocket ←→ ログ・進捗リアルタイム配信                   │
│  REST API  ←→ パイプライン制御 / SAM2 / プレビュー        │
│  HTML5 Canvas ← SAM2 クリック操作                        │
│  three.js (CDN) ← 3D プレビュー                          │
└────────────────────────────────────────────────────────┘
         ↕ HTTP / WebSocket (port 7860)
┌─ Docker Container ─────────────────────────────────────┐
│  FastAPI + Uvicorn                                      │
│  ├─ /api/pipeline/*  パイプライン制御                     │
│  ├─ /api/sam2/*      SAM2 操作 (click → mask PNG)       │
│  ├─ /api/preview/*   出力ファイルサービング                │
│  ├─ /ws              WebSocket (ログ + 進捗)             │
│  └─ /                静的ファイル (index.html)            │
│                                                          │
│  Pipeline Runner (asyncio.to_thread で各ステージ実行)     │
│  └─ 既存 stage_*.py をそのまま import して呼び出し         │
└────────────────────────────────────────────────────────┘
```

### トリプルフィルタ (Stage 3)

Pi3X は全画像に対して推論し、3段階のフィルタで対象物体の点群を抽出する:

1. **信頼度フィルタ** — Pi3X の出力信頼度が閾値未満の点を除去
2. **深度エッジフィルタ** — 深度の不連続箇所 (物体境界のアーティファクト) を除去
3. **SAM2 マスクフィルタ** — セグメンテーションマスク外の点を除去

### PyTorch / JAX 共存

DiffCD (JAX) と SAM2/Pi3X (PyTorch) は GPU コンテキストが競合するため、DiffCD はサブプロセスとして実行される。`XLA_PYTHON_CLIENT_PREALLOCATE=false` により JAX の事前メモリ確保を無効化し、PyTorch との共存を実現している。

### VRAM 管理

RTX 4090 (16GB) で全ステージを動作させるため、各 GPU ステージ終了時にモデルを明示的に解放し `torch.cuda.empty_cache()` でメモリを回収する。

Pi3X は OOM 時に以下の順で自動フォールバックする:

1. まず Pi3X 入力フレーム数を削減 (解像度は維持)
2. 次に `PIXEL_LIMIT` を縮小
3. 最後にチャンク推論へフォールバック

## VRAM とパフォーマンス

RTX 4090 Laptop (16GB) での実測値:

| ステージ | VRAM ピーク | 所要時間 |
|----------|-----------|---------|
| SAM2 (large) | ~2 GB | ~15秒 (伝播) |
| Pi3X (20フレーム, 150Kpx) | ~14 GB | ~1分 |
| DiffCD (res=384, 25Kバッチ) | ~10 GB | ~10分 |
| デノイズ / テクスチャ | CPU のみ | ~1分 |

> **注意**: 16GB GPU で品質を優先する場合、まず `MAX_FRAMES` を下げて `PIXEL_LIMIT` は高めに維持する。  
> 例: `MAX_FRAMES=20~28, PIXEL_LIMIT=220000~255000`。  
> さらに不足する場合のみ `PIXEL_LIMIT` を段階的に下げる。

## Docker イメージ構成

```
nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
├── Python 3.11 + system deps (ffmpeg, OpenGL)
├── PyTorch 2.5+ (CUDA 12.1 wheels)
├── Python deps (numpy<2.0, opencv, fastapi, uvicorn, scipy, trimesh, xatlas, ...)
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

まず `MAX_FRAMES` を下げ、`PIXEL_LIMIT` はできるだけ維持する。ダッシュボードの Advanced Settings で変更するか、CLI の場合:

```bash
docker compose run --rm --service-ports \
  --entrypoint python3 \
  -e MAX_FRAMES=20 -e PIXEL_LIMIT=240000 \
  pipeline /app/scripts/pipeline.py /data/input/video.mp4
```

### ダッシュボードに接続できない

`docker compose up` でコンテナが正常起動しているか確認:

```bash
docker compose logs --tail 20
# "Uvicorn running on http://0.0.0.0:7860" が表示されていれば OK
```

ポート 7860 が他のプロセスで使用されていないか確認:

```bash
lsof -i :7860
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
