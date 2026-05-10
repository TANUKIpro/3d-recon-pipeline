# Stage 4 — LiTo (ml-lito) reconstruction backend

clip2mesh の Stage 4 (3D 再構成) は、既定の **gs2mesh** に加えて Apple の
[**ml-lito**](https://github.com/apple/ml-lito) (LiTo: Surface Light Field
Tokenization, ICLR 2026) をバックエンドとして選択できる。LiTo は単一画像から
3D Gaussians を生成する image-to-3D モデルで、撮影条件が厳しいケースや、
Objaverse 系の物体カテゴリで形状プライアを活かしたいユースケースに向く。

> ⚠️ **研究目的限定**
> LiTo の事前学習済み重みは Apple ML Research Model License により
> **研究用途に限定**されている。商用利用 / 製品開発 / 製品/サービス組み込み
> は許可されていない。詳細は
> <https://github.com/apple/ml-lito/blob/main/LICENSE_MODEL>。
> 本バックエンドを起動するには、研究用途で使うことを了承する旨の環境変数
> `CLIP2MESH_ACCEPT_LITO_RESEARCH_LICENSE=1` を設定する必要がある。

---

## 切り替え方法

```bash
# CLI
CLIP2MESH_ACCEPT_LITO_RESEARCH_LICENSE=1 \
  python -m scripts.pipeline /data/input/video.mp4 \
    --output-dir /data/output \
    --reconstructor lito
```

ダッシュボード経由でも `Reconstructor` セレクタから `lito` を選べる。
研究用途同意フラグはコンテナの環境変数として渡す:

```bash
docker compose run \
  -e CLIP2MESH_ACCEPT_LITO_RESEARCH_LICENSE=1 \
  clip2mesh ...
```

## アーキテクチャ概要

```
COLMAP (Stage 2) ──→ camera_poses.json + sparse model
SAM2   (Stage 3) ──→ masks/*.png
                        ├──> [select frame]──> RGBA letterbox 518×518
                        │                         ↓
                        │                    [LiTo bridge: subprocess, /opt/ml-lito/.venv]
                        │                         ↓
                        │                    Gaussians PLY (canonical frame)
                        │                         ↓
                        │                    [render bridge → multi-view depth/rgb/alpha]
                        │                         ↓
                        │                    [tsdf_core.fuse_tsdf, gs2mesh と共有]
                        │                         ↓
                        │                    canonical mesh PLY
                        │                         ↓
                        └──> [Sim(3) alignment]──> p4_mesh/object_mesh.ply (world frame)
```

* **frame_selector**: SAM2 マスク被覆率 + Laplacian sharpness の複合スコアで 1 視点を選定。`mask coverage 5–80%`、bbox 短辺 ≥ 256 px、連結成分 1 個などの品質ゲートを通る。
* **mask_compositor**: 選定フレーム RGB と SAM2 マスクで `bbox + 0.8 fill_ratio` の正方クロップ → 透明背景 letterbox → 518×518 リサイズ。
* **lito_runner**: `/opt/ml-lito/.venv/bin/python` で `scripts/lito/bridge/lito_infer.py` を subprocess 起動。Gaussians PLY (LiTo canonical frame) を出力。
* **gaussian_to_mesh**: フィボナッチ球面 60 視点 + 入力視点を生成 → 別の bridge (`lito_render.py`) で gsplat レンダリング → 不透明度と入力視点との角度から confidence を計算 → `tsdf_core.fuse_tsdf` で TSDF Fusion → canonical mesh。
* **colmap_align**: 入力フレームに観測される COLMAP 3D 点を SAM2 マスクで前景フィルタ → LiTo Gaussian 中心とで PCA-init Sim(3) → Open3D point-to-point ICP で精細化 → `object_mesh.ply` を world frame で出力。

## 出力契約 (Stage 5 と互換)

`{output_dir}/p4_mesh/object_mesh.ply`
: PLY (頂点 x/y/z + 任意の RGB)。COLMAP 世界座標系。

`{output_dir}/p4_mesh/lito_workspace/`
: 中間成果物。
* `selected_frame.png` (518×518 RGBA)、`frame_score.json`、`compose_meta.json`
* `gaussians_canonical.ply` + `gaussians_canonical.meta.json` (LiTo 出力)
* `render_views.json` + `tsdf_views/` (per-view rgb/depth/alpha + summary.json)
* `mesh_canonical.ply` (アラインメント前)
* `alignment.json` (R, t, s, residual_rms, n_source_points, n_target_points)
* `LICENSE_RESEARCH_ONLY.txt` (生成物の派生扱いを明示)

Stage 5 (texture bake) は `object_mesh.ply` を gs2mesh 経路と同じ規約で消費するため、Stage 5 以降は無変更で動く。

## 重み (~8 GB)

`/data/models/lito/` 配下に置く:

| ファイル | サイズ | 用途 |
|---------|-------|------|
| `lito_dit_rgba.ckpt` | 6.86 GiB | image-to-3D 生成 (DiT) |
| `lito_new.ckpt` | 1.08 GiB | tokenizer (DiT が内部参照) |

ダウンロード方法:
* **ビルド時に取り込む**: `docker compose build --build-arg LITO_PREFETCH_WEIGHTS=1`
* **初回起動時に DL**: 何もしない (LiTo bridge が `/data/models/lito/` に取得しキャッシュ)

CDN: `https://ml-site.cdn-apple.com/models/lito/`

## VRAM / 時間

| 項目 | 値 (公式実測) | 備考 |
|------|-------------|------|
| 推論 VRAM | ~16 GB peak (H100) | clip2mesh の t16/t18 ティアで動作 |
| サンプリング | ~4.6 s (H100, 20 Heun steps, CFG 3.0) | `LITO_INFERENCE_STEPS` で変更可 |
| 多視点レンダリング | ~10–20 s (60 views, 512×512) | gsplat |
| TSDF Fusion + ICP | gs2mesh と同等 | tsdf_core を共有 |

合計で gs2mesh の 3〜5 割の総時間を見込む。実機は Phase 0c で計測。

## 既知の制約

* **多視点情報の破棄**: LiTo は単一画像入力のため、COLMAP の 3D 幾何制約は最終アラインメント時にしか使われない。多視点で観測されている部分は一見正確だが、裏面はモデルプライアによる外挿。
* **物体中心前提**: LiTo は object-centric シーンに最適化されている。シーン全体や複数物体の同時再構成には不向き。
* **OOD カテゴリ**: Objaverse / ObjaverseXL の分布外 (透明体、高周波鏡面、極端に大型 / 小型な物体) では形状崩壊しうる。frame_selector の品質ゲートでは検知しきれないため、出力メッシュは目視確認推奨。
* **アラインメント残差**: SAM2 マスクが甘い (背景を含む) と Sim(3) ICP が引き寄せられて世界座標が歪む。`p4_mesh/lito_workspace/alignment.json` の `residual_rms` を確認。

## 主要パラメータ (`scripts/config_defaults.py`)

| 名前 | 既定値 | 役割 |
|------|--------|------|
| `RECONSTRUCTOR` | `gs2mesh` | バックエンド選択 |
| `LITO_MODEL_NAME` | `lito_dit_rgba` | チェックポイント名 |
| `LITO_INFERENCE_STEPS` | `20` | Heun ODE ステップ数 |
| `LITO_CFG_SCALE` | `3.0` | CFG スケール |
| `LITO_FRAME_SELECTION_W` | `(0.5, 0.3, 0.2)` | (mask, triangulation, sharpness) の重み |
| `LITO_GATE_MIN_BBOX_SHORT_PX` | `256` | bbox 短辺の最小値 |
| `LITO_GATE_MIN_MASK_COVERAGE` | `0.05` | マスク被覆率の最小値 |
| `LITO_GATE_MAX_MASK_COVERAGE` | `0.80` | マスク被覆率の最大値 |
| `LITO_VENV_PYTHON` | `/opt/ml-lito/.venv/bin/python` | venv の Python |
| `LITO_BRIDGE_SCRIPT` | `/app/scripts/lito/bridge/lito_infer.py` | image-to-3D bridge |
| `LITO_SUBPROCESS_TIMEOUT_S` | `600` | bridge subprocess のタイムアウト |
| `LITO_ACCEPT_RESEARCH_LICENSE_ENV` | `CLIP2MESH_ACCEPT_LITO_RESEARCH_LICENSE` | 同意フラグの env 名 |

## 評価ランブック (Phase 5: Cereal で chamfer 比較)

`.claude/plans/lito_integration.md` §14 のベースラインに対し、
`scripts/eval_chamfer.py` で chamfer distance を計測する。受入基準は
**chamfer / gt_diagonal ≤ 0.30** (= 既存 gs2mesh ベースラインの 70%
以内)。

### 1. Docker ビルド + 重み (ユーザー実行)
```bash
docker compose build --build-arg LITO_PREFETCH_WEIGHTS=1
# あるいはビルド時間を短縮したい場合:
#   docker compose build  # weights are downloaded on first lito invocation
```

### 2. Cereal を lito 経路で完走
```bash
docker compose run --rm clip2mesh \
  env CLIP2MESH_ACCEPT_LITO_RESEARCH_LICENSE=1 \
  python -m scripts.pipeline \
    --video /data/input/Cereal.MOV \
    --reconstructor lito
```
SAM2 のインタラクティブ選択は CLI からは行えないため、ダッシュボード
(`docker compose up dashboard` → ブラウザで `Reconstructor=lito` を
選択) で実施するか、`p3_masks/` に既存マスクをコピーしてから
`--skip-to 4` を併用する。

### 3. Chamfer distance を測定
```bash
python -m scripts.eval_chamfer \
  --gt   /home/roboworks/repos/3d-recon-pipeline/data/output/objects/@main/Cereal/p6_cleanup/Cereal/textured_mesh_cleaned.obj \
  --pred /home/roboworks/repos/3d-recon-pipeline-work/data/output/objects/@<branch>/Cereal/p4_mesh/object_mesh.ply \
  --align icp \
  --threshold 0.30 \
  --out      data/output/objects/@<branch>/Cereal/p4_mesh/chamfer_vs_gs2mesh.json
```
exit code 0 = 受入基準クリア、1 = 未達。テクスチャベイク後の
`p6_cleanup/<obj>/textured_mesh_cleaned.obj` も同様に比較できる。

## 関連リンク

* 公式: <https://github.com/apple/ml-lito>
* プロジェクトページ: <https://apple.github.io/ml-lito/>
* Apple ML Research: <https://machinelearning.apple.com/research/lito>
* 設計計画書: `.claude/plans/lito_integration.md`
